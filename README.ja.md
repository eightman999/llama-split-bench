**English version → [README.md](README.md)**

# llama-split-bench

**マルチGPUマシンで、llama.cppの `--split-mode layer`(パイプライン分割・デフォルト)と `--split-mode tensor`(TP)のどちらが実際に速いのか**、そして2枚目以降のGPUがどれだけ効くのかを計測するツールです。

設定ファイル1つ、コマンド1つで、調整は不要です。各モードについて目的のコンテキストサイズまでの深度ラダーを実行し、prefill / decode スループット(投機デコードの採択率込み)を記録し、実プロンプト計測でdecodeを補正し、比較図(日本語・英語)を自動生成します。ワークフロー全体が非対話・機械可読で、人間と同様に**AIエージェントが操作することを前提に設計**しています(「AIエージェントに使わせる前提の設計」参照)。

![example](examples/example-2xv100-3way-ja.png)

*参考例: Tesla V100 ×2 (PG500-216 + V100-PCIE、PCIe 3.0 x8/x8、NVLinkなし)、Qwen3.8-27B UD-Q4_K_M、262kコンテキスト。結果: tensor分割は全深度でdecodeが勝ち(深さ0で +32% → 260kで +66%)。layer分割のdecodeは単一GPUと同一で、分割は「速度」ではなく「VRAM」を買うものだと分かります。*

## スコープ・対応環境・対象読者

**得られるもの** — マシンごとに1コマンドで: 実測に基づく答え(layer / tensor / 単一GPU)、比較図(日英)、そして他者が検証できる生の証跡(`run-info.json`、段ごとのJSON、サンプラログ)。

**ビルド・環境への依存** — 本ツールは `llama-server` バイナリと HTTP(`/completion` とその `timings` フィールド)しか使いません。SM(アーキテクチャ)固有のコードは一切なく、任意アーキテクチャ向けのCUDAビルドでも、他バックエンド(ROCm、Vulkan、Metal、CPU)でも動きます(バイナリは `bench.conf` の値)。NVIDIA限定の付加機能(外部プロセスガード、GPUサンプラ)は他環境では自動でオフに縮退し、投機デコードは任意(`SPEC_ARGS=""` で無効)。サーバの起動コマンドは完全に設定可能(FAQ参照)で、`LAUNCH_PREFIX` が `numactl`/`taskset`/`env` 等の前置に対応します。**GPU枚数は設定であってコードではありません:** 何枚でも動きます — `DEVICES=CUDA0,CUDA1,CUDA2,CUDA3` で4枚分割、`TENSOR_SPLIT` で不均等な重み比、`MODES`/`--mode-spec` で任意の構成、`--mode-spec "s1|CUDA1|"` のようなカード毎の単一GPUベースライン。**検証状況:** E2E実走はLinux + CUDA(sm70、V100×2)のみ。**マルチGPU(TP=2以上)の動作は一般には未確認です:** E2Eで確認できたのは参照環境の2枚構成だけで、それ以外のTP=2環境、およびTP=3以上(3〜5枚)の構成・他バックエンド・他アーキテクチャは同一コード経路ですが未検証です — まず4分のスモークを実行してください。

**対象読者** — 「このモデルをGPUに分割すべきか、どの方式か」に答える必要がある人: ワークステーション構築者、単機LLM運用者、そしてAIエージェント(次節)。数値は仕様として環境依存で、ユーザー間で一定なのは手順と証跡の形式です。

## AIエージェントに使わせる前提の設計

ワークフロー全体が非対話・機械可読で、エージェントが無人で回せるようになっています:

- **1コマンド+1設定ファイル** — 設定は `bench.conf`/`bench.local.conf` のみ。対話プロンプトもTUIもなし。
- **デタッチ前提** — 先に`mkdir -p runs`を一度実行し、`setsid nohup bash run-bench.sh <tag> … > runs/<tag>.log 2>&1 &`で起動してログをポーリング: `BENCH-DONE`(exit 0)=完了、`BENCH-ABORT: …`(exit 1)=計測失敗、`BENCH-FIGFAIL`(exit 1)=計測完了だが図の生成に失敗。SIGTERM/SIGINTはトラップされ、サーバとサンプラを掃除します(孤児を残しません)。
- **機械可読の結果** — `run-info.json`(バイナリsha256、モード/デバイス、ctx、KV型、起動プレフィックス)と段ごとの `results-*.json`。図はJSONだけから再生成できます(`plot_bench.py --dir …`、再計測不要)。
- **静的ゲート** — `check.sh`(bash -n、py_compile、任意でpyflakes)、タグ再利用の保護、計測対象GPUに他プロセスが乗ったら中断する外部プロセスガード。
- **決定的な手順** — 段構成・生成長・補正方法が固定なので、実行同士を比較できます(「なんとなく」ではなく)。

## 計測内容(この方式にした理由)

1. **深度ラダー(コンテキスト再利用)** — サーバには段階ごとに伸びる1本のプロンプトを送ります(0 → 32k → … → target)。各段で*増分*のprefill速度(`prompt_per_second`)を記録し、続けてNトークン(`ignore_eos`、既定1000)を生成した定常速度がdecode(`predicted_per_second`)です。実際のエージェント利用(長いプロンプト→深い位置での生成)を模した形です。
2. **深さ0のprefill(新規プロンプト)** — 512/2048/8192トークンのプロンプトを`cache_prompt=false`で送り、空コンテキストでの正しいprefill速度を測ります。(ラダー初段は11トークンのプロンプトなので、そのprefill値は計測上のアーティファクトで、作図には使いません。)
3. **投機デコード / MTP採択率** — `--spec-type draft-mtp` ではdecode速度がドラフトの採択数に依存します。ラダーの合成テキストはほぼ全部採択される(≈1.0 → 楽観側)ため、本ツールは別途「実プロンプト3本(ワークロード代理、temp 0.7 / top_p 0.9)」を1モードで実行し、補正係数(通常0.7〜0.9)を算出します。係数は*全モード・全深度に一律*で適用され、薄い破線の「実運用推定」として描かれます。decodeの絶対値は採択率に依存しますが、layer対tensorの相対比較は影響を受けません。
4. **実行ごとに残す証拠** — バイナリのsha256とバージョン、モード/デバイス表、KVキャッシュ型、段ごとの採択率、2秒間隔のGPUサンプラ(温度/電力、任意でファンPWM)、そして計測対象GPUに他プロセスが乗ったら中断する外部プロセスガード。

## クイックスタート

必要環境: Linux、そのllama.cppビルドが対応するGPU(1枚以上)、`python3`(計測は標準ライブラリのみ、作図に`matplotlib`)、`runs/`用の空き容量(1回数MB+任意の応答ダンプ)。GPUサンプラと外部プロセスガードは`nvidia-smi`と`CUDA<n>`形式のデバイス名を使います(他のバックエンドでは自動的に縮退: ガード無効・ファン記録オフ)。

```bash
git clone https://github.com/kuraneko1/llama-split-bench
cd llama-split-bench

# 初回のみ: 作図用のpython環境
python3 -m venv ~/.venvs/bench-plot
~/.venvs/bench-plot/bin/pip install matplotlib

# マシン固有の設定(git管理外)
cp bench.conf bench.local.conf
$EDITOR bench.local.conf        # MODEL(必須), MMPROJ(任意), DEVICES, CTX … を設定

./list-devices.sh               # llama.cppのデバイスとGPU構成を表示(llama-serverがPATHに無い場合はパスを引数で渡す)

# runs/はgit管理外なので、クローン直後に一度だけ作成
mkdir -p runs

# 本計測: layer + tensor(全DEVICES) + 単一GPUベースライン(3モード262kで約1時間)
setsid nohup bash run-bench.sh my-run-1 > runs/my-run-1.log 2>&1 &
# 完了マーカーを待つ(tail -fは終了しないので使わない)。マーカー:
#   BENCH-DONE=完了 | BENCH-ABORT: ...=失敗 | BENCH-FIGFAIL=図の生成に失敗
until grep -qE "BENCH-(DONE|ABORT|FIGFAIL)" runs/my-run-1.log; do sleep 30; done
tail -5 runs/my-run-1.log
```

図は生JSONと同じ場所に出ます: `runs/my-run-1/split-bench-ja.png` / `split-bench-en.png`。

まずスモーク(約4分、最小ラダーで全パイプライン検証):

```bash
bash run-bench.sh smoke --stages 0,4000 --n-predict 100 --ctx 8192 \
    --pp0-sizes 512,2048 --modes layer,single --no-real
```

1モードだけフル計測(腕の再現など):

```bash
bash run-bench.sh t1 --modes tensor
```

prefill/decode だけを計測したい場合(layer/tensor比較は不要):

```bash
bash run-bench.sh p1 --profile                              # 現在のDEVICES・既定splitで2パネル図
bash run-bench.sh p2 --mode-spec "myarm|CUDA0,CUDA1|tensor" # 名前と構成を自由に指定
```

![profile例](examples/example-profile-2xv100-ja.png)

*`--profile` の出力例(参照環境): 単一構成(tensor、262kラダー)のprefill/decodeを2パネルで作図。比較不要でも同じプロトコルが適用されます。薄破線=実運用推定。*

## 図の読み方

| パネル | 内容 |
|---|---|
| ① | prefill(depth-0は新規プロンプト実測値) |
| ② | decode — 実線=実測、**薄い破線=実運用推定**(実測 × 実プロンプト補正係数) |
| ③ | 先頭2系列の相対差(例: tensor vs layer) |
| ④ | ベースライン(既定=単一GPU)に対する伸び率。そのモードが実行に無ければ自動非表示 |

端点の数字は各パネル右端の内側に描かれます(太字=実測、細字=推定)。どの要素も再計測なしで切り替え可能です。**`--vs off`(実行全体に適用するなら`bench.conf`の`VS_PANEL=off`)で4つ目のパネルが消え、図が再配置されます**:

| オプション | フラグ / 設定 | 効果 |
|---|---|---|
| 対単一GPUパネル | `--vs off` · `VS_PANEL=off` (bench.conf) | **④が消える** — 単一GPUを測っていない場合、または比較が不要な場合 |
| 実運用推定線 | `--estimate off` · `--no-real` (実行時) | 薄い破線はその実行の実プロンプト補正係数から描かれ、**単一GPUの有無に関係なく、計測した全系列に表示**されます。`--estimate off` で非表示、`--no-real` なら補正係数そのものを計測しません |
| ベースライン | `--baseline <モード名>` · `BASELINE` (bench.conf) | ④が比較する相手(既定=`single`。そのモードが実行に無ければ自動非表示) |
| 系列の絞り込み | `--series layer,tensor` | 指定した実行だけを描画 |
| 言語 | `--lang ja` · `--lang en` | 日本語 / 英語の図 |

再計測なしの描き直し例:

```bash
~/.venvs/bench-plot/bin/python plot_bench.py --dir runs/my-run-1 \
    --series layer,tensor --lang en --vs off
```

## FAQ

- **結局 layer と tensor どっち?** 参考計測の2×V100環境では、decode重視のC=1配信なら全深度でtensor。layerは「VRAMを分けたい、decodeが単一GPU並でよい」場合のみ。交差点はマシンごとに違うはずで、それを測るのがこのツールです。
- **`--split-mode row` は?** 上流で非推奨のため計測対象外です。
- **`GGML_CUDA_P2P=1` は効く?** 参考環境では実質無効(内部AllReduceが既にx8で片方向~6.6GB/sを飽和、NCCL未リンク)。手元で試す価値はあります — 再実行は簡単です。
- **サーバが出す `NCCL not compiled` / `backend sampling ... CPU` 警告は?** 本ベンチには無害です。
- **なぜ各段1000トークン生成?** 定常decodeはMTP採択のばらつきを均すのに数百トークン必要なため。100では心もとなく、1000で図に載る精度になります。
- **モデルがQwenでない/MTP非対応の場合は?** `bench.conf`の`SPEC_ARGS`を空にすれば他は全てモデル非依存です。実プロンプト補正用のプロンプトは`measure_real.py --prompts-json`で差し替えできます(スクリプト参照)。
- **同じタグで再実行すると?** 仕様として拒否します(前回の`results-*-pp0.json`/`results-real.json`が新しい図に混入するため)。新しいタグを使ってください(`--reuse`は意図的な追記専用)。
- **単一GPUとの比較はいらない場合は?** 2通り: そのモードを実行しない(`--modes layer,tensor` — ④は自動非表示になり、単一GPU計測の時間も節約)、または`bench.conf`で`VS_PANEL=off`(単発の描き直しなら`--vs off`)。
- **サーバの起動コマンドは変えられる?** argvは`bench.conf`から組み立てます(`BIN`, `LAUNCH_PREFIX`, `MODEL`, `MMPROJ`, `DEVICES`/`MODES`, `NGL`, `THREADS`, `FA`, `JINJA`, `KV_K/V`, `SPEC_ARGS`, `SPEC_DEVICE`, `TENSOR_SPLIT`, `LOAD_MODE`, `CACHE_ARGS`, `HOST`, `PORT`, `EXTRA_ARGS`)。任意フラグは`EXTRA_ARGS`で追記、`numactl`/`taskset`/`env`等の前置は`LAUNCH_PREFIX`、ラッパースクリプトを使うなら`BIN`に指定。正確なserver argvは腕ごとに`argv-<モード名>.txt`へ記録され(`run-info.json`の`modes[].argv_file`)、バイナリのハッシュと併せて他者との比較の証跡になります。
- **1構成だけ計測したい(比較不要)場合は?** `--profile` で現在の `DEVICES` を1本の腕として計測し、prefill/decodeの2パネル図を生成します。`--mode-spec "名前|デバイス|split"`(複数指定可)で任意の腕を定義できます(例: `--mode-spec "gpu0|CUDA0|"` で単カード)。
- **runs/や図を公開する場合は?** `run-info.json`と図のタイトルには`MACHINE`ラベル(既定=ホスト名+GPU名)が埋め込まれます — 公開予定の計測では`MACHINE`を中立的な文字列に設定してください。`real-response-*.txt`、`responses-*/`、`server-*.log`にはプロンプト/出力やバックエンド情報が入り得ます — センシティブなら投稿前に削除を。
- **他人の設定やrunディレクトリを使う場合は?** `bench.conf`/`bench.local.conf`はbashでsourceされます(信頼できる設定のみ使用)。各実行の正確なserver argvは腕ごとに`argv-<モード名>.txt`へ記録され(`run-info.json`の`modes[].argv_file`)、バイナリのハッシュと併せて検証できます — 共有された結果を信じる前に確認を。

## ファイル構成

| ファイル | 役割 |
|---|---|
| `bench.conf` | 全設定(マシン固有値は`bench.local.conf`に、git管理外) |
| `run-bench.sh` | オーケストレータ: モードごとにサーバ起動 → ラダー → 深さ0 → 実プロンプト → run-info → 作図 |
| `measure_ladder.py` | 深度ラダー(コンテキスト再利用)、MTP採択率、外部プロセスガード |
| `measure_pp0.py` | 深さ0(新規プロンプト)のprefill計測 |
| `measure_real.py` | 実プロンプト補正計測 |
| `plot_bench.py` | 作図 — 流儀はコードに凍結(パネル構成、端点数字、トグル) |
| `list-devices.sh` | デバイス構成の確認用ヘルパ |
| `check.sh` | 静的チェック(bash -n、py_compile、任意でpyflakes)—コミット前に実行 |
| `runs/<tag>/` | 実行ごとの結果: 生JSON、サーバログ、サンプラログ、図 |

## ライセンス

MIT
