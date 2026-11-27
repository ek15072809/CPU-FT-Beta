/* CPU-FineTuning Web UI application logic (ja/en bilingual). v2 */
"use strict";

/* ------------------------------------------------------------------ */
/* i18n: labels, hints and TOOLTIPS for every parameter                */
/* ------------------------------------------------------------------ */
const I18N = {
  ja: {
    tabs: ["モデル", "データセット", "学習設定", "結果"],
    t_model_src: "モデルの指定",
    h_model_src: "HuggingFace のリポジトリID（例: organization/model-name）またはローカルフォルダのパス（例: C:\\models\\my-model）。パスは \" \" で囲まれていても自動的に剥がされます。すべてのキャッシュはプロジェクトフォルダ内に保存されます。",
    btn_model_info: "情報取得",
    l_dtype: "演算精度（マスター重み）",
    l_device: "実行デバイス",
    supported: "対応アーキテクチャ（同じアーキテクチャならパラメータ数が違っても対応）: LFM2 / LFM2.5 / LFM3 / Qwen3.5 / Gemma4 系",
    l_datacache: "使用する構築済みデータセット",
    h_datacache: "複数の構築済みキャッシュがある場合に、学習で使うものを明示的に選べます。「自動」は設定から決まる名前のキャッシュだけを使い、なければ新規構築します。選択したキャッシュは設定に保存されます。",
    t_events: "主なイベント（最良更新・悪化・LR低減・再開）",
    t_plots: "学習グラフ（PNG）",
    t_metrics_help: "指標の説明",
    t_result_log: "学習ログ（末尾）",
    plot_lr: "学習率", plot_loss: "学習損失", plot_grad: "勾配ノルム", plot_val: "検証CE",
    t_chart_lr: "学習率の推移", t_chart_grad: "勾配ノルムの推移",
    t_data_sources: "データソース",
    h_data_sources: "HuggingFace データセットID またはローカルファイル (json / jsonl / csv / tsv / parquet / arrow / txt / md) を複数指定できます。ローカルのパスは \" \" で囲まれていてもOK（右クリックの「パスのコピー」対応）。チャット形式 (messages / conversations)、Alpaca形式、prompt/response、素テキスト(PT用)を自動判別します。",
    btn_add_source: "＋ ソース追加",
    btn_preview: "全ソースをプレビュー",
    btn_remove: "削除",
    src_source: "ソース（HF ID またはパス）",
    src_split: "split",
    src_name: "config名",
    src_format: "形式（空=自動）",
    src_weight: "混合比",
    src_subset: "先頭N件",
    t_raw_preview: "生データ（先頭行）",
    t_norm_preview: "正規化後の最初の例（入力 / 出力）",
    t_data_build: "データセット構築（パッキング）",
    l_ctx: "コンテキスト長（1ステップの最大トークン数）",
    l_eps: "1ステップあたりの例の数（0=自動）",
    l_maxseq: "最大シーケンス数（0=無制限）",
    l_val_ratio: "検証データ比率",
    l_min_chars: "最小文字数",
    l_dedup: "重複除去",
    l_seed: "シード",
    l_order: "データの使い方",
    order_front: "先頭から順に（小データを前半に混合）",
    order_shuffle: "全体をシャッフル",
    h_order: "「先頭から順に」はデータをシャッフルせず、先頭から順に学習します。複数ソースでは、最大ソースの先頭N件（「先頭N件」指定）に小さいソースをランダムに混合します。学習を途中で打ち切っても小さいデータを先に使い切りたい場合に有効です。",
    l_longdoc: "長文構造学習（汎用）",
    l_longdoc_min: "長文の最小文字数",
    l_longdoc_sent: "長文の最小文数",
    l_longdoc_max: "長文サンプル上限",
    l_longdoc_tail: "学習する後半文字数",
    l_longdoc_prefix: "文書プラン最大トークン",
    btn_build: "データセット構築",
    building: "構築中...",
    build_done: "構築完了",
    build_error: "構築エラー",
    t_cached: "構築済みデータセット",
    t_examples_preview: "構築データの最初の例（入力 / 出力）",
    t_order_vis: "実際のデータ順（先頭20件）",
    t_order_examples: "例の順序",
    t_order_sequences: "学習シーケンスの順序",
    t_order_legend: "凡例",
    t_train_basic: "学習の基本設定",
    l_task: "タスク",
    l_preset: "プリセット",
    l_steps: "総ステップ数",
    l_lr_sci: "学習率（指数表記）",
    l_lr_dec: "学習率（小数表記）",
    l_wd: "Weight Decay",
    l_sched: "スケジューラ",
    l_warmup: "ウォームアップ（ステップ）",
    l_eval: "検証間隔（ステップ）",
    l_runname: "実行名",
    t_limits: "学習の打ち切り条件",
    l_maxhours: "時間制限（時間・0=無効）",
    l_maxepochs: "エポック制限（0=無効）",
    h_limits: "時間制限は学習開始からの経過時間で、エポック制限は全学習シーケンスを1周=1エポックとして計算されます。総ステップ数=0 のときは打ち切り条件に従って学習し、総ステップ数・時間制限・エポック制限がすべて0なら1エポックで終了します。",
    t_guard: "検証ECガード（最良モデル・停止・自動再開）",
    l_guardmode: "悪化したときの動作（1つだけ選択）",
    l_stopreg: "検証EC悪化で自動停止",
    l_exportbest: "最良ECモデルを自動エクスポート",
    l_autorestart: "悪化時に最良ECから自動再開",
    l_patience: "悪化判定の連続回数",
    l_mindelta: "改善とみなす最小ΔCE",
    l_restartmode: "再開LRの下げ方",
    l_restartdecay: "指数モードのLR倍率",
    l_restartfloor: "再開LRの下限",
    l_maxrestarts: "最大再開回数（0=無制限）",
    l_probe: "学習プローブ（トークン）",
    opt_restart_harmonic: "1/n（調和）",
    opt_restart_exponential: "倍率（指数）",
    h_guard: "検証ECが最良値を更新すると、その時点のチェックポイントを保存します。悪化判定は「最良値＋最小ΔCE」を超えた状態が指定回数続いたときです。モードは1つだけ選べます: 再開=最良点へ戻してLRを下げ再試行 / 停止=最良点で停止 / 無効=巻き戻さず学習を続行（最良モデルの保存・エクスポートは有効なまま）。矛盾した旧設定は保存時に自動的に整合されます。",
    t_estimate: "所要時間の推定",
    btn_estimate: "推定する",
    estimate_note: "モデルのパラメータ数・データ数・設定から概算します（実際の速度はCPU負荷で変動します）。",
    t_spike: "勾配スパイク対策",
    l_spike_mode: "動作モード",
    l_spike_pct: "検出しきい値（%超過）",
    l_spike_win: "比較ウィンドウ（ステップ）",
    h_spike: "直近ウィンドウの平均勾配ノルムから指定%を超えて増加したステップを異常と判定します。「スキップ」はそのステップの重み更新を行いません。「縮小」は更新量を通常ステップと同じ水準に縮めます。両者は排他的で、ウィンドウが溜まる最初の数ステップは判定を行いません。",
    t_ascot_adv: "ASCOT 詳細設定（上級者向け）",
    h_ascot: "num_spans を増やすと品質が上がり速度が下がります。mode=global は目的関数が厳密、local は最速の代理損失、auto は自動選択です。",
    d_span: "1つの学習スパン（アンカー）のトークン長。大きいほど1スパンで長い文脈を学習しますが、1ステップの計算量が増えます。既定128。",
    d_spans: "1ステップでサンプリングするスパン数。多いほど品質↑・速度↓（品質と速度の主なトレードオフ）。既定5。",
    d_prefix: "スパン直前のトークン数。凍結パスでここを先行計算し、スパン開始時の内部状態（conv/KV）を移植に使います。v7では長文でもこの局所文脈を必ず残します。64〜128が目安。",
    d_plan: "長文サンプルで文書プラン（見出し・重要事実・繰り返し語）に使うトークン数。v7は [文書先頭 | プラン | 直前の局所文脈] を並べるため、プランと通常の次トークン予測を両立します。0で無効。",
    d_headlr: "tied LMヘッド（埋め込み兼用）の学習率倍率。ヘッドは毎ステップ更新されるため、下げると出力分布と入力埋め込みが安定します。0.05〜0.15を推奨。",
    d_sink: "アテンションシンクとして文書先頭から確保するトークン数。長コンテキストの安定化に使います。16前後で十分です。",
    d_slice: "1ステップで学習する連続レイヤー数（層ルーレット）。大きいほど1ステップのカバレッジ↑・計算量↑。2〜6を推奨。",
    d_mode: "global=スライスより上の層も勾配計算に含め厳密な目的関数（高品質・低速）。local=スライス内だけの代理損失（最速）。auto=状況に応じ自動選択。",
    d_head: "スパン内で損失を計算する位置数の上限。0で全位置。絞ると速くなりますが、サンプリング誤差が増えます。64前後がバランス良好。",
    d_sample: "層（スライス開始位置）の選び方。balanced=全開始位置を均等に巡回（v7推奨）、adaptive=勾配の大きい層を優先、uniform=均等ランダム、top=上位層を優先。v6のadaptiveは下位層に偏り、上位層が取り残される原因になりました。",
    d_refresh: "凍結側int8キャッシュの更新頻度。1=毎ステップ更新（厳密）。大きいほど高速化されますが、参照する重みが古くなります。2〜4が実用的。",
    d_floor: "重要度サンプリングに混ぜる一様分布の割合。大きいほど全ブロックに学習機会が広がり、小さいほど重要な箇所に集中します。0.25前後を推奨。",
    d_block: "重要度スコアリングのブロックサイズ。スパン選択の細かさを決めます。通常は変更不要。",
    d_clip: "勾配ノルムの絶対上限（クリッピング）。1.0が標準的。発散気味なら小さくします。",
    t_log_ckpt: "ログとチェックポイント",
    l_logevery: "表示間隔（ステップ）",
    l_ckpt_every: "チェックポイント間隔",
    l_ckpt_keep: "保持する最新数",
    l_ckpt_mile: "マイルストーン保存（nステップ毎）",
    l_ckpt_opt: "オプティマイザ状態の保存",
    l_gn: "勾配ノルム表示",
    l_mem: "メモリ表示",
    l_export_name: "エクスポート名",
    l_autoexport: "終了時に自動エクスポート",
    l_deltadtype: "チェックポイントの差分形式",
    l_plots: "PNGグラフを保存",
    h_ckpt: "チェックポイントは元モデルとの差分（int8など）で保存するため、モデル全体の約2倍の容量を毎回消費しません。エクスポート時に完全な重みへ復元されます。PNGグラフは runs/<実行名>/plots/ に保存されます。",
    t_cli_export: "CLIコマンドのエクスポート",
    h_cli_export: "現在の学習設定を、Web UIを起動せずに実行できるPowerShellコマンドとして出力します。コピーまたは .ps1 として保存し、run.py があるプロジェクトのルートで実行してください。",
    btn_cli_export: "コマンドを生成",
    btn_cli_copy: "コピー",
    btn_cli_download: ".ps1 として保存",
    cli_empty: "「コマンドを生成」を押すと、ここに --フラグ形式のコマンドが表示されます。",
    cli_copied: "CLIコマンドをコピーしました",
    cli_downloaded: "CLIコマンドを .ps1 として保存しました",
    start_note: "開始すると Web UI は自動的に終了します。学習の経過は新しく開くコンソール（コマンドプロンプト）に表示され、終了後もログを確認できるようコンソールは開いたままになります（Enterで閉じます）。終了後にこのUIが自動で再起動し、結果・PNGグラフ・エクスポートを確認できます。",
    btn_start: "学習開始",
    l_resume: "チェックポイントから再開",
    t_results: "学習結果",
    btn_refresh: "更新",
    t_chart_loss: "損失",
    t_chart_val: "検証CE",
    t_gen: "生成テスト",
    gen_select: "モデルを選択（最近使用した順）",
    gen_base: "学習前",
    gen_trained: "学習後",
    l_gen_compare: "学習前後を比較（順次実行）",
    h_gen: "パスを直接入力するか、学習で使ったモデル／学習後のモデルをサジェストから選べます。比較はベースモデル→学習後モデルの順に1つずつ読み込むため、2モデル同時にメモリへ載せることはありません。",
    btn_gen: "生成",
    col_role: "役割",
    col_input: "入力",
    col_output: "出力",
    role_user: "ユーザー",
    role_assistant: "アシスタント",
    role_system: "システム",
    saved: "保存しました",
    started: "学習を開始しました。このページは自動で閉じます。",
    err: "エラー",
    no_data: "データなし",
  },
  en: {
    tabs: ["Model", "Dataset", "Training", "Results"],
    t_model_src: "Model source",
    h_model_src: "HuggingFace repo id (e.g. organization/model-name) or a local folder path (e.g. C:\\models\\my-model). Quoted paths are unwrapped automatically. All caches stay inside the project folder.",
    btn_model_info: "Get info",
    l_dtype: "Compute dtype (master weights)",
    l_device: "Execution device",
    supported: "Supported architectures (any parameter count within the same arch): LFM2 / LFM2.5 / LFM3 / Qwen3.5 / Gemma4 families",
    l_datacache: "Use an existing built dataset",
    h_datacache: "When several caches exist, choose exactly which one training should use. Auto uses only the cache name derived from the current settings and builds it when missing. The selection is saved in the config.",
    t_events: "Major events (best / worsening / LR reduction / restart)",
    t_plots: "Training graphs (PNG)",
    t_metrics_help: "Metric descriptions",
    t_result_log: "Training log (tail)",
    plot_lr: "Learning rate", plot_loss: "Training loss", plot_grad: "Gradient norm", plot_val: "Validation CE",
    t_chart_lr: "Learning-rate history", t_chart_grad: "Gradient-norm history",
    t_data_sources: "Data sources",
    h_data_sources: "HuggingFace dataset ids or local files (json / jsonl / csv / tsv / parquet / arrow / txt / md); mix multiple sources. Quoted local paths are fine (Windows 'Copy as path'). Chat (messages / conversations), Alpaca, prompt/response and raw text (PT) formats are auto-detected.",
    btn_add_source: "+ Add source",
    btn_preview: "Preview all sources",
    btn_remove: "Remove",
    src_source: "Source (HF id or path)",
    src_split: "split",
    src_name: "config",
    src_format: "format (auto)",
    src_weight: "weight",
    src_subset: "first N",
    t_raw_preview: "Raw data (first rows)",
    t_norm_preview: "First normalized example (input / output)",
    t_data_build: "Dataset build (packing)",
    l_ctx: "Context length (max tokens per step)",
    l_eps: "Examples per step (0 = auto)",
    l_maxseq: "Max sequences (0 = unlimited)",
    l_val_ratio: "Validation ratio",
    l_min_chars: "Min chars",
    l_dedup: "Dedup",
    l_seed: "Seed",
    l_order: "Data order",
    order_front: "From the front (mix small data into the prefix)",
    order_shuffle: "Shuffle everything",
    h_order: "'From the front' keeps source order and mixes smaller sources randomly into the first N examples of the largest source (set 'first N' on it). Use this when you may stop before the full dataset is consumed but still want the small dataset fully used.",
    l_longdoc: "Long-document structure learning (generic)",
    l_longdoc_min: "Min long-doc chars",
    l_longdoc_sent: "Min long-doc sentences",
    l_longdoc_max: "Max long-doc samples",
    l_longdoc_tail: "Tail chars to train",
    l_longdoc_prefix: "Max plan tokens",
    btn_build: "Build dataset",
    building: "Building...",
    build_done: "Done",
    build_error: "Build error",
    t_cached: "Built datasets",
    t_examples_preview: "First built examples (input / output)",
    t_order_vis: "Actual data order (first 20)",
    t_order_examples: "Example order",
    t_order_sequences: "Training sequence order",
    t_order_legend: "Legend",
    t_train_basic: "Training basics",
    l_task: "Task",
    l_preset: "Preset",
    l_steps: "Total steps",
    l_lr_sci: "Learning rate (scientific)",
    l_lr_dec: "Learning rate (decimal)",
    l_wd: "Weight decay",
    l_sched: "Scheduler",
    l_warmup: "Warmup steps",
    l_eval: "Eval every (steps)",
    l_runname: "Run name",
    t_limits: "Stop conditions",
    l_maxhours: "Time limit (hours, 0 = off)",
    l_maxepochs: "Epoch limit (0 = off)",
    h_limits: "The time limit counts from training start; one epoch = one pass over all training sequences. With total steps = 0, training follows the stop conditions; if total steps, time limit and epoch limit are all 0, it stops after exactly one epoch.",
    t_guard: "Validation guard (best model / stop / auto-restart)",
    l_guardmode: "Action on regression (single choice)",
    l_stopreg: "Stop on validation regression",
    l_exportbest: "Export the best-CE model",
    l_autorestart: "Restart from best on regression",
    l_patience: "Consecutive bad evaluations",
    l_mindelta: "Minimum CE improvement",
    l_restartmode: "Restart LR schedule",
    l_restartdecay: "Exponential-mode LR factor",
    l_restartfloor: "Restart LR floor",
    l_maxrestarts: "Max restarts (0 = unlimited)",
    l_probe: "Training probe tokens",
    opt_restart_harmonic: "1/n (harmonic)",
    opt_restart_exponential: "Factor (exponential)",
    h_guard: "A checkpoint is saved whenever validation CE reaches a new best. Regression means CE stayed above best+min-delta for the configured number of evaluations. Choose exactly one mode: Restart = roll back to the best step, lower LR and retry; Stop = stop at the best step; Off = keep training without rollback (best-model save/export still works). Contradictory legacy flags are normalized when the config is saved.",
    t_estimate: "Estimated time",
    btn_estimate: "Estimate",
    estimate_note: "Rough estimate from model parameters, dataset size and settings (actual speed varies with CPU load).",
    t_spike: "Gradient spike handling",
    l_spike_mode: "Mode",
    l_spike_pct: "Threshold (% above average)",
    l_spike_win: "Window (steps)",
    h_spike: "A step whose grad norm exceeds the recent-window average by more than the threshold is treated as a spike. 'Skip' drops that update; 'Scale' rescales it to a typical step. The modes are mutually exclusive; the first few steps (until the window fills) are never flagged.",
    t_ascot_adv: "ASCOT advanced settings",
    h_ascot: "More num_spans = better quality, slower. mode=global is the exact objective; local is the fastest surrogate; auto picks automatically.",
    d_span: "Token length of one anchor span. Larger spans learn longer contexts per span but cost more compute per step. Default 128.",
    d_spans: "Anchor spans sampled per step. More = better quality, slower (the main quality/speed trade-off). Default 5.",
    d_prefix: "Tokens before each span processed by the frozen pass to build the transplanted conv/KV state. v7 always keeps this local context for long documents too. More = better context fidelity, more compute. 64-128 is typical.",
    d_plan: "Tokens used for the document plan (headings, key facts, recurring terms) in long-document samples. v7 uses [document start | plan | immediate local context], so the plan and ordinary next-token prediction coexist. 0 = off.",
    d_headlr: "Learning-rate multiplier for the tied LM head (shared with embeddings). Lower it to keep the output distribution and input embeddings stable. 0.05-0.15 recommended.",
    d_sink: "Tokens kept from the document start as the attention sink. Stabilizes long contexts; around 16 is enough.",
    d_slice: "Consecutive layers trained per step (layer roulette). Larger = more coverage per step, more compute. 2-6 recommended.",
    d_mode: "global = layers above the slice join the gradient pass for the exact objective (quality, slower). local = surrogate loss inside the slice only (fastest). auto = automatic.",
    d_head: "Upper limit of loss positions inside a span (0 = all). Fewer positions are faster but add sampling noise. Around 64 is balanced.",
    d_sample: "How slice (layer) start positions are chosen: balanced (round-robin over every start; v7 default), adaptive (gradient-mass biased), uniform, or top-biased. v6's adaptive sampler starved the upper layers.",
    d_refresh: "How often the frozen int8 caches are requantized. 1 = every step (exact); larger = faster but the frozen pass sees older weights. 2-4 is practical.",
    d_floor: "Uniform mixture of the importance sampling. Larger spreads training over all blocks; smaller concentrates on important ones. ~0.25 recommended.",
    d_block: "Block size of the importance scoring (granularity of span selection). Usually no change needed.",
    d_clip: "Absolute gradient-norm clipping limit. 1.0 is standard; lower it if training looks unstable.",
    t_log_ckpt: "Logging & checkpoints",
    l_logevery: "Display interval (steps)",
    l_ckpt_every: "Checkpoint every",
    l_ckpt_keep: "Keep last",
    l_ckpt_mile: "Milestone every",
    l_ckpt_opt: "Save optimizer state",
    l_gn: "Show grad norm",
    l_mem: "Show memory",
    l_export_name: "Export name",
    l_autoexport: "Auto-export on finish",
    l_deltadtype: "Checkpoint delta format",
    l_plots: "Save PNG graphs",
    h_ckpt: "Checkpoints store differences from the original model (int8 by default), so they do not consume a full ~2x model copy every time. Full weights are reconstructed on export. PNG graphs are saved under runs/<run>/plots/.",
    t_cli_export: "Export CLI command",
    h_cli_export: "Generate a PowerShell command that reproduces the current training settings without starting the Web UI. Copy it or save it as a .ps1 file, then run it from the project folder containing run.py.",
    btn_cli_export: "Generate command",
    btn_cli_copy: "Copy",
    btn_cli_download: "Save as .ps1",
    cli_empty: "Click Generate command to show a --flag command here.",
    cli_copied: "CLI command copied",
    cli_downloaded: "CLI command saved as .ps1",
    start_note: "When training starts, this Web UI shuts down. Progress is shown in the new console window, which stays open after the run so the log remains visible (press Enter to close). When finished, this UI relaunches automatically for results, PNG graphs and the export.",
    btn_start: "Start training",
    l_resume: "Resume from checkpoint",
    t_results: "Results",
    btn_refresh: "Refresh",
    t_chart_loss: "Loss",
    t_chart_val: "Validation CE",
    t_gen: "Generation test",
    gen_select: "Select a model (most recently used first)",
    gen_base: "Before training",
    gen_trained: "After training",
    l_gen_compare: "Compare before/after (sequential)",
    h_gen: "Type a path directly, or pick a model used for training / a trained export from the suggestions. Comparison loads the base model first, frees it, then loads the tuned model, so both are never in memory at once.",
    btn_gen: "Generate",
    col_role: "Role",
    col_input: "Input",
    col_output: "Output",
    role_user: "User",
    role_assistant: "Assistant",
    role_system: "System",
    saved: "Saved",
    started: "Training started. This page will close.",
    err: "Error",
    no_data: "no data",
  },
};

/* tooltips (title attributes) per input id */
const TIPS = {
  ja: {
    "model-source": "HuggingFaceのリポジトリID（organization/model-name）か、ローカルにダウンロード済みのモデルフォルダのパスを入力します。",
    "model-device": "autoは CUDA → MPS → CPU の順で利用可能なデバイスを選びます。cpuを明示すると従来どおりCPUだけで実行します。cuda:N はCLI/configで指定できます。",
    "model-dtype": "autoは選択デバイスに合わせます。CPUでは速度とRAMからfp32/bf16、CUDAではbf16対応GPUならbf16・未対応（T4など）ならfp32、MPSではfp32を選びます。float16はCUDA向けの明示指定です。",
    "d-ctx-sel": "1ステップで処理するトークン数の上限。大きいほど1ステップの情報量が増えますが時間がかかります。",
    "d-ctx": "1ステップで処理するトークン数（512の倍数を推奨）。",
    "d-eps": "1つの学習ステップに詰め込むデータ（例）の件数。0で自動（コンテキスト長いっぱいまで詰める）。例: 4 とすると1ステップ=4件のデータになります。",
    "d-maxseq": "データセット構築で作るシーケンス数の上限。大きなデータセットで時間・メモリを使いすぎないための安全弁。0で無制限。",
    "d-val": "構築データから検証用に取り分ける割合（0.02 = 2%）。",
    "d-minch": "これより短いデータ（例）は除外します。",
    "d-dedup": "同一内容のデータ（例）を自動除去します。",
    "d-seed": "データのシャッフルなどに使う乱数シード。",
    "d-order": "先頭から順に=シャッフルせず、大きいデータの先頭N件に小さいデータを混合します。全体をシャッフル=従来のランダム順。",
    "d-cache": "構築済みキャッシュがある場合に、学習で使うものを明示的に選びます。「自動」は現在のデータ設定から決まる名前だけを使い、なければ新規構築します。選択したキャッシュは設定に保存され、設定と指紋が違ってもそのまま使われます（ログに警告）。",
    "d-longdoc": "長い文書（記事・Markdown・書籍・レポート・コード解説・HTMLなど）から「文書プラン（見出し・重要事実・繰り返し語/class）」と「後半領域」を対にした学習サンプルを自動生成します。後半の参照箇所を追加ブーストし、長文の書き切り・離れた情報の参照・用語/表記の一貫性を汎用的に学習します。推論速度は変わりません。",
    "d-longdoc-min": "これ以上の文字数を持つ文書だけを長文構造サンプルの対象にします。HTML専用ではありません。",
    "d-longdoc-sent": "見出しや改行が少ない文書でも、この文数以上あれば長文として扱います。",
    "d-longdoc-max": "生成する長文構造サンプルの上限。0で無制限。",
    "d-longdoc-tail": "文書末尾から学習する文字数。長いほど中間〜末尾を広くカバーします。",
    "d-longdoc-prefix": "文書プラン（見出し・重要事実・繰り返し語）に使う最大トークン数。ASCOTのprefix長に収まる範囲で使います。",
    "t-task": "SFT=指示チューニング（対話データ）。PT=継続事前学習（生テキスト）。",
    "t-preset": "speed=最速、balanced=速度と品質のバランス（推奨）、quality=最高品質。custom=手動設定。",
    "t-steps": "学習の総ステップ数。1ステップ=1シーケンス（例の詰め合わせ）の学習。0にすると打ち切り条件（時間/エポック）に従い、すべて0なら1エポックで終了します。",
    "t-lr-sci": "学習率（5e-6 のような指数表記）。小数表記欄と自動同期します。プリセットの目安は balanced=5e-6、quality=2e-6、speed=1e-5 です。ガードが「再開」のとき、検証EC悪化後は最良点から設定した方式で学習率を下げます。",
    "t-lr-dec": "学習率（0.000005 のような小数表記）。指数表記欄と自動同期します。",
    "t-guardmode": "検証ECガードの動作を1つだけ選びます。再開=最良チェックポイントへ戻って学習率を下げ再試行、停止=最良チェックポイントを残して終了、無効=そのまま最後まで学習。矛盾する組み合わせは保存時に自動で整合します。",
    "t-stopreg": "検証ECがベースラインより悪い、または最良値から悪化した状態が指定回数続いたら発動します。「ガードの動作」で停止/再開/無効を選びます。",
    "t-exportbest": "学習終了時に、最後のモデルではなく検証ECが最も良かったチェックポイントを書き出します（真の最小値。val_min_deltaでは変わりません）。",
    "t-autorestart": "「再開」モードの説明です。検証ECの悪化を検出したら最良ECチェックポイントへ戻り、学習率を下げて続きから再開します。再開回数の上限は0でLR下限までです。",
    "t-patience": "最良ECを更新しない検証が何回続いたら悪化と判定するか。1なら毎回すぐ、2以上ならノイズを少し待ちます。",
    "t-mindelta": "悪化・改善とみなすCEの最小幅。これ未満の改善では連続悪化カウントをリセットしません。ただし最良チェックポイントは常に真の最小値で更新されます。",
    "t-restartmode": "1/n（調和）は元の学習率を 1/2, 1/3, 1/4, … に下げます。倍率（指数）は元の学習率に係数を繰り返し掛けます（0.5なら1/2, 1/4, 1/8, …）。",
    "t-restartdecay": "指数モードで学習率へ掛ける倍率（0.5 = 半分ずつ）。調和モードでは使いません。",
    "t-restartfloor": "再試行で次に使う学習率がこの値を下回る場合は、無限ループを避けるため停止します。既定 1e-8。実用上は 1e-7 前後で十分なことが多いです。",
    "t-maxrestarts": "再試行の上限回数。0はLR下限に達するまで無制限。再開自体を無効にするにはガードの動作を「停止」または「無効」にします。",
    "t-probe": "毎回の検証で同じ学習先頭トークン列を全層フォワードし、サンプリング損失とは別の実CEを記録します。0で無効。",
    "t-wd": "重み減衰（正則化）。0.01が標準的。",
    "t-sched": "学習率のスケジュール。cosine=徐々に減衰（推奨）、linear=直線減衰、constant=固定。",
    "t-warmup": "学習率を0から目的値まで徐々に上げるステップ数。学習初期の破壊を防ぎます。",
    "t-eval": "何ステップごとに検証CEを計算するか。0で無効。",
    "t-runname": "この学習実行の名前。チェックポイントや履歴の保存先フォルダ名になります。",
    "t-maxhours": "この時間を超えたら学習を打ち切ります（例: 1.5 = 1時間半）。0で無効。",
    "t-maxepochs": "全学習シーケンスを1周=1エポックとして、この回数に達したら打ち切ります。0で無効。",
    "t-spikemode": "勾配ノルムが直近平均から急増したステップへの対処。無効/スキップ/縮小から選択。",
    "t-spikepct": "直近平均の何%増加を異常とみなすか（100 = 2倍で検出）。",
    "t-spikewin": "異常判定の基準にする直近ステップ数。",
    "a-span": "1つの学習スパンのトークン長。",
    "a-spans": "1ステップでサンプリングするスパン数。多いほど品質↑・速度↓。",
    "a-prefix": "スパン直前の状態計算に使うトークン数。多いほど文脈精度↑。",
    "a-sink": "文書先頭（アテンションシンク）として確保するトークン数。",
    "a-slice": "1ステップで学習する連続する層の数。",
    "a-mode": "global=厳密な目的関数（推奨）。local=最速の代理損失。auto=自動選択。",
    "a-head": "スパンごとに損失計算する位置数。0ですべての位置。",
    "a-sample": "層の選び方。adaptive=勾配の大きい層を優先、uniform=均等、top=上位層優先。",
    "a-refresh": "凍結側int8キャッシュの更新頻度。大きいほど速いが若干古い重みを参照。",
    "a-floor": "重要度サンプリングの一様フロア。大きいほど均等なサンプリング。",
    "a-block": "重要度スコアリングのブロックサイズ。",
    "a-clip": "勾配ノルムの絶対上限（クリッピング）。",
    "c-logevery": "コンソールにログを表示するステップ間隔。",
    "c-every": "何ステップごとにチェックポイントを保存するか。0で無効。",
    "c-keep": "ローリングで保持する最新チェックポイント数。",
    "c-mile": "このステップ数の倍数をマイルストーンとして削除せず残します。0で無効。",
    "c-opt": "ONにすると8-bit Adamの状態も保存し、正確に再開できます（ディスク増）。",
    "c-deltadtype": "チェックポイントは元モデルとの差分だけを保存します。int8は最小（既定）、bfloat16/float16は高精度、float32は完全一致に近い再現。復元・エクスポート時に完全なウェイトへ戻ります。",
    "c-plots": "学習終了時に学習率・学習損失・勾配ノルム・検証CE・全体概要のPNGを runs/<run>/plots に書き出します。",
    "c-gn": "コンソールログに勾配ノルムを表示。",
    "c-mem": "コンソールログにメモリ使用量を表示。",
    "c-exportname": "自動エクスポート先のフォルダ名（exports/ の下）。既存フォルダがある場合は日時を付けて上書きしません。",
    "c-autoexport": "学習終了時に完全なウェイトを自動で書き出します（推奨）。",
    "gen-model": "生成テストに使うモデルの場所（例: exports/finetuned）。学習run名（例: run1）も書き出し済みモデルに自動解決されます。",
    "gen-model-suggest": "学習で使ったモデルと学習後のモデルを、最後に使った順に表示します。選ぶとパス欄に反映されます。",
    "gen-compare": "チェックすると、ベースモデルで生成→メモリ解放→学習後モデルで生成、の順に実行します。",
    "gen-prompt": "モデルへの質問・指示を入力。",
  },
  en: {
    "model-source": "HuggingFace repo id (organization/model-name) or a path to a locally downloaded model folder.",
    "model-device": "auto selects the first available device in CUDA -> MPS -> CPU order. Choosing cpu keeps the previous CPU-only behavior. cuda:N can be set in the CLI/config.",
    "model-dtype": "auto follows the selected device: CPU picks fp32/bf16 from speed and RAM, CUDA uses bf16 when the GPU supports it (fp32 otherwise, e.g. T4), and MPS uses fp32. float16 is an explicit CUDA-oriented option.",
    "d-ctx-sel": "Upper limit of tokens processed per training step.",
    "d-ctx": "Tokens per step (multiples of 512 recommended).",
    "d-eps": "Number of data examples packed into one training step. 0 = auto (fill the context window).",
    "d-maxseq": "Cap on the number of packed sequences; a safety valve for big datasets. 0 = unlimited.",
    "d-val": "Fraction of built data held out for validation (0.02 = 2%).",
    "d-minch": "Examples shorter than this are dropped.",
    "d-dedup": "Automatically remove duplicate examples.",
    "d-seed": "Random seed for shuffling etc.",
    "d-order": "Front = no shuffle; mix smaller sources into the first N examples of the largest source. Shuffle = the previous global random order.",
    "d-cache": "Choose which previously built dataset cache training should use. Auto uses only the cache name derived from the current data settings (and builds a new one if absent). An explicit choice is saved in the config and is used even when its fingerprint differs (the log warns).",
    "d-longdoc": "Automatically creates long-document samples from any long text (articles, Markdown, books, reports, code documentation, HTML): a document plan (headings, key facts, recurring terms/classes) plus a later region. Later references to the plan get extra sampling weight, training long-form completion, distant reference use, and terminology/format consistency. Inference speed is unchanged.",
    "d-longdoc-min": "Only documents at least this long become long-document samples. This is not HTML-specific.",
    "d-longdoc-sent": "Even without headings/line structure, documents with at least this many sentences are treated as long documents.",
    "d-longdoc-max": "Maximum number of long-document samples (0 = unlimited).",
    "d-longdoc-tail": "Characters trained from the document's later region. Larger values cover more of the middle/end.",
    "d-longdoc-prefix": "Maximum tokens for the document plan (headings, key facts, recurring terms), bounded by ASCOT's prefix budget.",
    "t-task": "SFT = instruction tuning (chat data). PT = continued pre-training (raw text).",
    "t-preset": "speed = fastest, balanced = recommended default, quality = best per step, custom = manual.",
    "t-steps": "Total training steps. One step trains on one packed sequence. 0 = follow the stop conditions (time/epoch); if all are 0, training ends after one epoch.",
    "t-lr-sci": "Learning rate in scientific notation (e.g. 5e-6); syncs with the decimal field. Preset guidance: balanced=5e-6, quality=2e-6, speed=1e-5. With guard mode 'restart', validation regression retries the best point at a lower LR.",
    "t-lr-dec": "Learning rate in decimal notation (e.g. 0.000005); syncs with the scientific field.",
    "t-guardmode": "Choose exactly one validation-CE guard behavior. Restart = roll back to the best checkpoint and retry with a lower LR; stop = keep the best checkpoint and end; off = train to the normal stop condition. Contradictory legacy flags are normalized on save.",
    "t-stopreg": "Triggers when validation CE is worse than the baseline or stays worse than the best for the configured patience. Choose stop/restart/off with Guard behavior.",
    "t-exportbest": "Export the checkpoint with the true lowest validation CE instead of the last model. val_min_delta never changes this choice.",
    "t-autorestart": "This describes 'restart' mode: on validation regression, roll back to the best-CE checkpoint, lower the LR, and continue. A retry limit of 0 means retry down to the LR floor.",
    "t-patience": "How many consecutive evaluations without a new best count as regression. 1 acts immediately; 2+ waits for noise to settle.",
    "t-mindelta": "Minimum CE change treated as meaningful. Improvements smaller than this do not reset the consecutive-worse counter, but the best checkpoint still always follows the true minimum.",
    "t-restartmode": "1/n (harmonic) uses base/2, base/3, base/4, ... on retries. Factor (exponential) uses base*decay, base*decay^2, ... (0.5 -> 1/2, 1/4, 1/8, ...).",
    "t-restartdecay": "LR factor for exponential mode (0.5 = halve each retry); unused in harmonic mode.",
    "t-restartfloor": "Stop retrying when the next LR would be below this value. Default 1e-8; about 1e-7 is often enough in practice.",
    "t-maxrestarts": "Maximum retries; 0 = unlimited until the LR floor. To disable retries, set Guard behavior to stop or off.",
    "t-probe": "On every validation, evaluate the same fixed training token prefix with a full forward to log a real train CE separately from the sampled span loss. 0 = off.",
    "t-wd": "Weight decay (regularization); 0.01 is typical.",
    "t-sched": "LR schedule: cosine decay (recommended), linear decay, or constant.",
    "t-warmup": "Steps used to ramp the LR up from zero at the start.",
    "t-eval": "Compute validation CE every N steps (0 = off).",
    "t-runname": "Name of this run; becomes the folder for checkpoints and history.",
    "t-maxhours": "Stop training after this many hours (e.g. 1.5). 0 = off.",
    "t-maxepochs": "One epoch = one pass over all training sequences. Stop after N epochs. 0 = off.",
    "t-spikemode": "Action when a step's grad norm spikes above the recent average.",
    "t-spikepct": "How far above the recent average counts as a spike (100 = 2x).",
    "t-spikewin": "Number of recent steps used as the baseline.",
    "a-span": "Token length of one anchor span.",
    "a-spans": "Spans sampled per step; more = better quality, slower.",
    "a-prefix": "Tokens of local context used to build the transplanted state.",
    "a-sink": "Document-start tokens kept as the attention sink.",
    "a-slice": "Number of consecutive layers trained per step.",
    "a-mode": "global = exact objective (recommended); local = fastest surrogate; auto = automatic.",
    "a-head": "Loss positions per span (0 = all).",
    "a-sample": "Layer selection: adaptive (gradient-mass biased), uniform, or top-biased.",
    "a-refresh": "How often the frozen int8 caches are rebuilt; larger = faster but slightly stale.",
    "a-floor": "Uniform floor of the importance sampling.",
    "a-block": "Block size for importance scoring.",
    "a-clip": "Absolute gradient-norm clipping limit.",
    "c-logevery": "Console log interval in steps.",
    "c-every": "Save a checkpoint every N steps (0 = off).",
    "c-keep": "Rolling number of newest checkpoints to keep.",
    "c-mile": "Keep every Nth checkpoint forever as a milestone (0 = off).",
    "c-opt": "Also save the 8-bit Adam state for exact resumption (more disk).",
    "c-deltadtype": "Checkpoints store only the difference from the original model. int8 is smallest (default), bfloat16/float16 preserve more precision, float32 is nearly exact. Loading/exporting reconstructs full weights.",
    "c-plots": "At the end of training, write PNG graphs for LR, training loss, grad norm, validation CE, and an overview to runs/<run>/plots.",
    "c-gn": "Show grad norm in the console log.",
    "c-mem": "Show memory usage in the console log.",
    "c-exportname": "Folder name under exports/ for the automatic export. An existing non-empty folder is never overwritten (a timestamp is added).",
    "c-autoexport": "Write full weights automatically when training ends (recommended).",
    "gen-model": "Model location for the generation test (e.g. exports/finetuned). A training run name such as run1 is resolved to its exported model.",
    "gen-model-suggest": "Models used for training and trained exports, most recently used first. Selecting one fills the path field.",
    "gen-compare": "Runs base model generation, frees it, then runs the tuned model (never both at once).",
    "gen-prompt": "Question or instruction for the model.",
  },
};

/* Every number shown in the results panel has a plain-language explanation. */
const METRIC_HELP = {
  ja: {
    run_steps: "実際に完了した学習ステップ数。再開時は再試行で巻き戻した分を差し引いた、現在の履歴の長さです。",
    planned_steps: "設定から決まる当初の学習ステップ上限。時間制限で先に終了した場合はこれより小さくなります。",
    schedule_total: "学習率スケジュールが終端（最小LR）に到達する予定ステップ。時間制限がある場合は実測速度で補正されます。",
    stop_reason: "終了理由。time_limit=時間制限、epoch_limit=エポック上限、step_limit=ステップ上限、val_regression=検証悪化、restart_lr_floor=再開LRが下限未満、one_epoch_default=条件未指定の1エポック。",
    loss: "学習損失（サンプリングしたスパンの次トークンCE）。ノイズがあるため、最初と最後の数ステップ平均とEMAも併記します。",
    loss_window: "loss first→last の平均に使ったステップ数。短い実行で窓が重ならないよう全体の1/3（最大5）です。",
    loss_ema: "学習損失の指数移動平均。単発の損失より傾向が見やすい指標です。",
    loss_min: "学習中に記録された最小の学習損失（サンプリング推定値）。",
    val_baseline: "学習開始前（step 0）の検証CE（左）と、右側は最後に記録された検証CEです。小さいほど良い。",
    val_best: "検証CEの最小値（真の最小）とそのステップ。bestエクスポートはこの重みを使います。",
    val_last: "最後に記録された検証CE。bestより大きい場合は悪化しています。",
    val_final: "学習終了時に改めて評価した検証CE（最後の検証と同じステップなら再利用）。",
    val_delta: "ベースライン検証CE − 最後の検証CE。正なら改善、負なら悪化。",
    probe: "学習データ先頭の固定トークン列を全層フォワードしたCE。検証ではなく学習データ上の参考値なので、検証CEと直接比較しないでください。",
    grad_mean: "記録された勾配ノルムの平均。急増はスパイク、持続的な増大は不安定化の目安です。",
    grad_max: "記録された勾配ノルムの最大値。スパイク対策のしきい値調整に使います。",
    spikes: "勾配スパイクとして検出され、スキップまたは縮小されたステップ数。",
    sec_step: "1学習ステップあたりの平均時間（検証・保存を含む実測）。",
    steps_hour: "1時間あたりの学習ステップ数（平均）。",
    tokens_sec: "定常状態の1秒あたり処理トークン数（コンテキスト長÷平均秒）。",
    guard: "検証ECガードの設定。mode=restart/stop/off、patience=悪化判定の連続回数、minΔ=悪化/改善とみなす最小CE差（bestの選択には使いません）。",
    restarts: "悪化により最良点へ戻してLRを下げた回数と、各再試行のLR。",
    restart_policy: "再開時のLRの下げ方。harmonic=1/2,1/3,1/4…、exponential=倍率^回数。floor未満になったら停止します。",
    best_checkpoint: "検証CEが最小だったチェックポイント。差分形式で保存され、エクスポート時に完全な重みへ復元されます。",
    exported: "実際に書き出したモデルの元ステップと検証CE。bestエクスポートでは最終ステップと異なることがあります。",
    export_path: "完全な重みを書き出したフォルダ。",
    device: "使用したデバイスと演算精度。autoはCUDA→MPS→CPUの順で選択します。",
    dataset: "使用した構築済みデータセットのキャッシュ名。",
    cache: "使用した構築済みデータセットのキャッシュ名。",
    history: "学習履歴の保存先。",
  },
  en: {
    run_steps: "Number of completed training steps in the current history (rewound steps after a retry are removed).",
    planned_steps: "Original step limit derived from the settings. A time limit can stop the run earlier.",
    schedule_total: "Step at which the LR schedule reaches its minimum. With a time limit it is corrected from measured throughput.",
    stop_reason: "Why training ended: time_limit, epoch_limit, step_limit, val_regression, restart_lr_floor, or one_epoch_default.",
    loss: "Training loss (next-token CE on sampled spans). Because it is noisy, first/last window averages and EMA are shown too.",
    loss_window: "Number of steps in the first→last averages (one third of the run, capped at 5, so short runs do not overlap).",
    loss_ema: "Exponential moving average of training loss; easier to read than individual noisy steps.",
    loss_min: "Lowest recorded sampled training loss.",
    val_baseline: "Validation CE before training (step 0, left) and the last recorded validation CE (right); lower is better.",
    val_best: "True minimum validation CE and its step. Best export uses these weights.",
    val_last: "Last recorded validation CE. A value above best means regression.",
    val_final: "Validation CE evaluated at the end (reused when the last validation was at the final step).",
    val_delta: "Baseline validation CE minus last validation CE. Positive = improvement, negative = regression.",
    probe: "Full-forward CE on a fixed prefix of training data. This is a training-data reference, not held-out validation; do not compare it directly with validation CE.",
    grad_mean: "Mean recorded gradient norm. Spikes indicate instability; a sustained rise is also a warning.",
    grad_max: "Maximum recorded gradient norm; useful for tuning the spike threshold.",
    spikes: "Steps detected as gradient spikes and skipped or scaled.",
    sec_step: "Average wall-clock seconds per training step, including validation and saves.",
    steps_hour: "Training steps per hour (average).",
    tokens_sec: "Steady-state tokens processed per second (context length / average seconds).",
    guard: "Validation guard: mode=restart/stop/off, patience=consecutive bad evaluations, minΔ=minimum CE change treated as meaningful (never used to pick best).",
    restarts: "Number of rollbacks with LR reduction, and the new LR of each retry.",
    restart_policy: "Restart LR policy: harmonic=1/2,1/3,1/4..., exponential=factor^retry. Stops below the floor.",
    best_checkpoint: "Checkpoint with the lowest validation CE, stored as a delta and reconstructed on export.",
    exported: "Step and validation CE of the exported weights; best export can differ from the final step.",
    export_path: "Folder containing the reconstructed full model.",
    device: "Device and compute dtype. auto chooses CUDA → MPS → CPU when available.",
    dataset: "Built dataset cache used by this run.",
    cache: "Built dataset cache used by this run.",
    history: "Saved training history path.",
  },
};

let LANG = "ja";
let STATE = null;
const $ = (id) => document.getElementById(id);
const T = (k) => (I18N[LANG][k] !== undefined ? I18N[LANG][k] : k);
const TIP = (id) => (TIPS[LANG][id] || "");

function toast(msg, isErr) {
  const el = $("toast");
  el.textContent = msg;
  el.className = "show" + (isErr ? " error" : "");
  setTimeout(() => (el.className = ""), 3000);
}

async function api(path, body) {
  const opt = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const r = await fetch(path, opt);
  const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error || r.status);
  return j;
}

/* ------------------------------------------------------------------ */
function applyI18n() {
  document.documentElement.lang = LANG;
  document.querySelectorAll("#tabs button").forEach((b, i) => (b.textContent = T("tabs")[i]));
  /* NOTE: ids use hyphens in HTML; the v1 bug (underscore keys) is fixed */
  const map = {
    "t-model-src": "t_model_src", "h-model-src": "h_model_src",
    "btn-model-info": "btn_model_info", "l-dtype": "l_dtype",
    "l-device": "l_device",
    "l-datacache": "l_datacache", "h-datacache": "h_datacache",
    "t-data-sources": "t_data_sources", "h-data-sources": "h_data_sources",
    "btn-add-source": "btn_add_source", "btn-preview": "btn_preview",
    "t-raw-preview": "t_raw_preview", "t-norm-preview": "t_norm_preview",
    "t-data-build": "t_data_build", "l-ctx": "l_ctx", "l-eps": "l_eps",
    "l-maxseq": "l_maxseq", "l-val-ratio": "l_val_ratio", "l-min-chars": "l_min_chars",
    "l-dedup": "l_dedup", "l-seed": "l_seed", "btn-build": "btn_build",
    "l-order": "l_order", "l-longdoc": "l_longdoc", "h-order": "h_order",
    "l-longdoc-min": "l_longdoc_min", "l-longdoc-sent": "l_longdoc_sent",
    "l-longdoc-max": "l_longdoc_max", "l-longdoc-tail": "l_longdoc_tail",
    "l-longdoc-prefix": "l_longdoc_prefix",
    "t-cached": "t_cached", "t-train-basic": "t_train_basic", "l-task": "l_task",
    "l-preset": "l_preset", "l-steps": "l_steps", "l-lr-sci": "l_lr_sci",
    "l-lr-dec": "l_lr_dec", "l-wd": "l_wd", "l-sched": "l_sched",
    "l-warmup": "l_warmup", "l-eval": "l_eval", "l-runname": "l_runname",
    "t-limits": "t_limits", "l-maxhours": "l_maxhours", "l-maxepochs": "l_maxepochs",
    "h-limits": "h_limits", "t-guard": "t_guard", "l-guardmode": "l_guardmode",
    "l-stopreg": "l_stopreg",
    "l-exportbest": "l_exportbest", "l-autorestart": "l_autorestart",
    "l-patience": "l_patience", "l-mindelta": "l_mindelta",
    "l-restartmode": "l_restartmode",
    "opt-restart-harmonic": "opt_restart_harmonic",
    "opt-restart-exp": "opt_restart_exponential",
    "l-restartdecay": "l_restartdecay",
    "l-restartfloor": "l_restartfloor", "l-maxrestarts": "l_maxrestarts",
    "l-probe": "l_probe", "h-guard": "h_guard", "t-estimate": "t_estimate",
    "btn-estimate": "btn_estimate", "estimate-note": "estimate_note",
    "t-spike": "t_spike", "l-spike-mode": "l_spike_mode",
    "l-spike-pct": "l_spike_pct", "l-spike-win": "l_spike_win", "h-spike": "h_spike",
    "t-ascot-adv": "t_ascot_adv", "h-ascot": "h_ascot", "t-log-ckpt": "t_log_ckpt",
    "l-logevery": "l_logevery", "l-ckpt-every": "l_ckpt_every",
    "l-ckpt-keep": "l_ckpt_keep", "l-ckpt-mile": "l_ckpt_mile",
    "l-ckpt-opt": "l_ckpt_opt", "l-gn": "l_gn", "l-mem": "l_mem",
    "l-export-name": "l_export_name", "l-autoexport": "l_autoexport",
    "l-deltadtype": "l_deltadtype", "l-plots": "l_plots", "h-ckpt": "h_ckpt",
    "t-cli-export": "t_cli_export", "h-cli-export": "h_cli_export",
    "btn-cli-export": "btn_cli_export", "btn-cli-copy": "btn_cli_copy",
    "btn-cli-download": "btn_cli_download",
    "start-note": "start_note", "btn-start": "btn_start", "l-resume": "l_resume",
    "t-results": "t_results", "btn-refresh": "btn_refresh",
    "t-events": "t_events", "t-plots": "t_plots",
    "t-metrics-help": "t_metrics_help", "t-result-log": "t_result_log",
    "t-chart-loss": "t_chart_loss", "t-chart-val": "t_chart_val",
    "t-chart-lr": "t_chart_lr", "t-chart-grad": "t_chart_grad",
    "t-gen": "t_gen", "btn-gen": "btn_gen",
    "l-gen-compare": "l_gen_compare", "h-gen": "h_gen",
    "t-gen-base": "gen_base", "t-gen-tuned": "gen_trained",
  };
  for (const [id, key] of Object.entries(map)) {
    const el = $(id);
    if (el) el.textContent = T(key);
  }
  const cli = $("cli-command");
  if (cli && !cli.dataset.command) cli.textContent = T("cli_empty");
  const orderSel = $("d-order");
  if (orderSel) {
    orderSel.options[0].text = T("order_front");
    orderSel.options[1].text = T("order_shuffle");
  }
  $("supported-models").textContent = T("supported");
  const heads = ["src_source", "src_split", "src_name", "src_format", "src_weight", "src_subset"];
  $("sources-header").innerHTML = heads.map((k) => `<th>${T(k)}</th>`).join("") + "<th></th>";
  document.querySelectorAll("#lang-ja,#lang-en").forEach((b) => (b.className = ""));
  $("lang-" + LANG).className = "active";
  /* tooltips for every known input */
  for (const id of Object.keys(TIPS[LANG])) {
    const el = $(id);
    if (el) el.title = TIP(id);
  }
  /* visible descriptions for the ASCOT advanced settings */
  const AD = {
    "ad-span": "d_span", "ad-spans": "d_spans", "ad-prefix": "d_prefix",
    "ad-plan": "d_plan", "ad-sink": "d_sink", "ad-slice": "d_slice",
    "ad-mode": "d_mode", "ad-head": "d_head", "ad-headlr": "d_headlr",
    "ad-sample": "d_sample", "ad-refresh": "d_refresh",
    "ad-floor": "d_floor", "ad-block": "d_block", "ad-clip": "d_clip",
  };
  for (const [id, key] of Object.entries(AD)) {
    const el = $(id);
    if (el) el.textContent = T(key);
  }
  applyGuardUi();
}

/* ------------------------------------------------------------------ */
/* learning-rate dual format (scientific <-> decimal)                   */
/* ------------------------------------------------------------------ */
let lrSyncing = false;

function fmtSci(v) {
  if (!isFinite(v) || v === 0) return "0";
  const exp = Math.floor(Math.log10(Math.abs(v)));
  const mant = v / Math.pow(10, exp);
  const m = (Math.abs(mant - 1) < 1e-12) ? 1 : Math.round(mant * 1000) / 1000;
  return `${m}e${exp}`;
}
function fmtDec(v) {
  if (!isFinite(v)) return "";
  return String(parseFloat(v.toPrecision(6)));
}
function bindLr() {
  $("t-lr-sci").addEventListener("input", () => {
    if (lrSyncing) return;
    const v = parseFloat($("t-lr-sci").value);
    if (isFinite(v)) { lrSyncing = true; $("t-lr-dec").value = fmtDec(v); lrSyncing = false; }
  });
  $("t-lr-dec").addEventListener("input", () => {
    if (lrSyncing) return;
    const v = parseFloat($("t-lr-dec").value);
    if (isFinite(v)) { lrSyncing = true; $("t-lr-sci").value = fmtSci(v); lrSyncing = false; }
  });
}

/* ------------------------------------------------------------------ */
function readConfig() {
  const specs = [];
  document.querySelectorAll("#sources-table tbody tr").forEach((tr) => {
    const g = (c) => tr.querySelector(`td:nth-child(${c}) input, td:nth-child(${c}) select`).value;
    const source = g(1).trim();
    if (!source) return;              // ignore an unused blank row
    specs.push({
      source, split: g(2).trim() || "train", name: g(3).trim() || null,
      format: g(4).trim() || null, weight: parseFloat(g(5)) || 1.0,
      subset: g(6).trim() ? parseInt(g(6)) : null,
    });
  });
  const guardMode = $("t-guardmode").value || "restart";
  const minDeltaText = String($("t-mindelta").value ?? "").trim();
  const minDelta = minDeltaText === "" ? 0.001 : parseFloat(minDeltaText);
  return {
    model_source: $("model-source").value.trim(),
    model_dtype: $("model-dtype").value,
    device: $("model-device").value || "auto",
    dataset_cache: ($("d-cache").value || "").replace(/\.pt$/i, ""),
    dataset_specs: specs,
    data: {
      context_length: parseInt($("d-ctx").value) || 32768,
      examples_per_step: parseInt($("d-eps").value) || 0,
      max_sequences: parseInt($("d-maxseq").value) || 0,
      val_ratio: parseFloat($("d-val").value) || 0.02,
      min_chars: parseInt($("d-minch").value) || 0,
      dedup: $("d-dedup").value === "true",
      seed: parseInt($("d-seed").value) || 0,
      dataset_order: $("d-order").value,
      long_doc_structure: $("d-longdoc").value === "true",
      long_doc_min_chars: parseInt($("d-longdoc-min").value) || 1200,
      long_doc_min_sentences: parseInt($("d-longdoc-sent").value) || 4,
      long_doc_max_docs: parseInt($("d-longdoc-max").value) || 0,
      long_doc_tail_chars: parseInt($("d-longdoc-tail").value) || 16000,
      long_doc_prefix_tokens: parseInt($("d-longdoc-prefix").value) || 64,
    },
    task: $("t-task").value,
    ascot: {
      span_length: +$("a-span").value, num_spans: +$("a-spans").value,
      prefix_len: +$("a-prefix").value, sink_len: +$("a-sink").value,
      long_plan_tokens: +$("a-plan").value,
      slice_size: +$("a-slice").value, mode: $("a-mode").value,
      head_positions: +$("a-head").value,
      head_lr_scale: +$("a-headlr").value,
      slice_sampling: $("a-sample").value,
      shadow_refresh_every: +$("a-refresh").value,
      uniform_floor: +$("a-floor").value, block_size: +$("a-block").value,
      max_grad_norm: +$("a-clip").value,
    },
    lr: parseFloat($("t-lr-sci").value) || 5e-6,
    weight_decay: parseFloat($("t-wd").value),
    scheduler: $("t-sched").value,
    warmup_steps: +$("t-warmup").value,
    total_steps: +$("t-steps").value,
    eval_every: +$("t-eval").value,
    guard_mode: guardMode,
    stop_on_val_regression: guardMode !== "off",
    export_best: $("t-exportbest").value === "true",
    auto_restart: guardMode === "restart",
    val_patience: Math.max(1, +$("t-patience").value || 1),
    val_min_delta: Number.isFinite(minDelta) ? Math.max(0, minDelta) : 0.001,
    restart_lr_mode: $("t-restartmode").value || "harmonic",
    restart_lr_decay: parseFloat($("t-restartdecay").value) || 0.5,
    restart_lr_floor: Number.isFinite(parseFloat($("t-restartfloor").value))
      ? parseFloat($("t-restartfloor").value) : 1e-8,
    max_restarts: +$("t-maxrestarts").value || 0,
    train_probe_tokens: +$("t-probe").value || 0,
    run_name: $("t-runname").value.trim() || "run1",
    grad_spike_mode: $("t-spikemode").value,
    grad_spike_threshold_pct: parseFloat($("t-spikepct").value) || 100,
    grad_spike_window: +$("t-spikewin").value || 20,
    max_hours: parseFloat($("t-maxhours").value) || 0,
    max_epochs: +$("t-maxepochs").value || 0,
    log_every: +$("c-logevery").value,
    show_grad_norm: $("c-gn").value === "true",
    show_memory: $("c-mem").value === "true",
    lang: LANG,
    checkpoint_every: +$("c-every").value,
    checkpoint_keep_last: +$("c-keep").value,
    checkpoint_keep_every: +$("c-mile").value,
    checkpoint_optimizer: $("c-opt").value === "true",
    checkpoint_delta_dtype: $("c-deltadtype").value || "int8",
    save_plots: $("c-plots").value === "true",
    auto_export: $("c-autoexport").value === "true",
    export_name: $("c-exportname").value.trim() || "finetuned",
  };
}

function writeConfig(cfg) {
  if (!cfg) return;
  const cfgDevice = cfg.device || "auto";
  $("model-device").value = ["auto", "cpu", "cuda", "mps"].includes(cfgDevice)
    ? cfgDevice : "auto";
  $("model-source").value = cfg.model_source || "";
  $("model-dtype").value = cfg.model_dtype || "auto";
  const ctx = (cfg.data && cfg.data.context_length) || 32768;
  const sel = $("d-ctx-sel");
  const has = Array.from(sel.options).some((o) => +o.value === ctx);
  sel.value = has ? String(ctx) : "custom";
  $("d-ctx").style.display = has ? "none" : "block";
  $("d-ctx").value = ctx;
  $("d-eps").value = (cfg.data && cfg.data.examples_per_step) || 0;
  $("d-maxseq").value = (cfg.data && cfg.data.max_sequences) != null ? cfg.data.max_sequences : 4000;
  $("d-val").value = (cfg.data && cfg.data.val_ratio) || 0.02;
  $("d-minch").value = (cfg.data && cfg.data.min_chars) || 10;
  $("d-dedup").value = String(!!(cfg.data && cfg.data.dedup));
  $("d-seed").value = (cfg.data && cfg.data.seed) || 0;
  $("d-order").value = (cfg.data && cfg.data.dataset_order) || "front";
  $("d-longdoc").value = String(cfg.data ? cfg.data.long_doc_structure !== false : true);
  $("d-longdoc-min").value = (cfg.data && cfg.data.long_doc_min_chars) || 1200;
  $("d-longdoc-sent").value = (cfg.data && cfg.data.long_doc_min_sentences) || 4;
  $("d-longdoc-max").value = (cfg.data && cfg.data.long_doc_max_docs) != null ? cfg.data.long_doc_max_docs : 512;
  $("d-longdoc-tail").value = (cfg.data && cfg.data.long_doc_tail_chars) || 16000;
  $("d-longdoc-prefix").value = (cfg.data && cfg.data.long_doc_prefix_tokens) || 64;
  $("t-task").value = cfg.task || "sft";
  $("a-span").value = cfg.ascot?.span_length ?? 128;
  $("a-spans").value = cfg.ascot?.num_spans ?? 4;
  $("a-prefix").value = cfg.ascot?.prefix_len ?? 48;
  $("a-plan").value = cfg.ascot?.long_plan_tokens ?? 64;
  $("a-sink").value = cfg.ascot?.sink_len ?? 16;
  $("a-slice").value = cfg.ascot?.slice_size ?? 4;
  $("a-mode").value = cfg.ascot?.mode ?? "global";
  $("a-head").value = cfg.ascot?.head_positions ?? 64;
  $("a-headlr").value = cfg.ascot?.head_lr_scale ?? 0.10;
  $("a-sample").value = cfg.ascot?.slice_sampling ?? "balanced";
  $("a-refresh").value = cfg.ascot?.shadow_refresh_every ?? 4;
  $("a-floor").value = cfg.ascot?.uniform_floor ?? 0.25;
  $("a-block").value = cfg.ascot?.block_size ?? 512;
  $("a-clip").value = cfg.ascot?.max_grad_norm ?? 1.0;
  const lr = cfg.lr ?? 5e-6;
  $("t-lr-sci").value = fmtSci(lr);
  $("t-lr-dec").value = fmtDec(lr);
  $("t-wd").value = cfg.weight_decay ?? 0.01;
  $("t-sched").value = cfg.scheduler || "cosine";
  $("t-warmup").value = cfg.warmup_steps ?? 20;
  $("t-steps").value = cfg.total_steps ?? 0;
  $("t-eval").value = cfg.eval_every ?? 20;
  $("t-runname").value = cfg.run_name || "run1";
  $("t-spikemode").value = cfg.grad_spike_mode || "off";
  $("t-spikepct").value = cfg.grad_spike_threshold_pct ?? 100;
  $("t-spikewin").value = cfg.grad_spike_window ?? 20;
  $("t-maxhours").value = cfg.max_hours ?? 0;
  $("t-maxepochs").value = cfg.max_epochs ?? 1;
  let guardMode = cfg.guard_mode;
  if (!["restart", "stop", "off"].includes(guardMode)) {
    guardMode = cfg.auto_restart !== false ? "restart"
      : (cfg.stop_on_val_regression !== false ? "stop" : "off");
  }
  $("t-guardmode").value = guardMode;
  $("t-exportbest").value = String(cfg.export_best !== false);
  $("t-patience").value = cfg.val_patience ?? 1;
  $("t-mindelta").value = cfg.val_min_delta ?? 0.001;
  $("t-restartmode").value = cfg.restart_lr_mode || "harmonic";
  $("t-restartdecay").value = cfg.restart_lr_decay ?? 0.5;
  $("t-restartfloor").value = cfg.restart_lr_floor ?? 1e-8;
  $("t-maxrestarts").value = cfg.max_restarts ?? 0;
  $("t-probe").value = cfg.train_probe_tokens ?? 512;
  $("c-logevery").value = cfg.log_every ?? 1;
  $("c-gn").value = String(cfg.show_grad_norm !== false);
  $("c-mem").value = String(cfg.show_memory !== false);
  $("c-every").value = cfg.checkpoint_every ?? 100;
  $("c-keep").value = cfg.checkpoint_keep_last ?? 2;
  $("c-mile").value = cfg.checkpoint_keep_every ?? 500;
  $("c-opt").value = String(!!cfg.checkpoint_optimizer);
  $("c-deltadtype").value = cfg.checkpoint_delta_dtype || "int8";
  $("c-plots").value = String(cfg.save_plots !== false);
  $("c-autoexport").value = String(cfg.auto_export !== false);
  $("c-exportname").value = cfg.export_name || "finetuned";
  $("d-cache").dataset.wanted = String(cfg.dataset_cache || "").replace(/\.pt$/i, "");
  renderSources((cfg.dataset_specs || []).filter(
    (s) => String((s && s.source) || "").trim()));
  refreshRuns();
  refreshDatasets();
  applyGuardUi();
}

/* ------------------------------------------------------------------ */
let configSaveTimer = null;

function applyGuardUi() {
  const modeEl = $("t-guardmode");
  if (!modeEl) return;
  const mode = modeEl.value || "restart";
  const exp = ($("t-restartmode").value || "harmonic") === "exponential";
  const optText = LANG === "ja"
    ? ["再開（LRを下げて最良点から再試行）", "停止（最良点で停止）",
       "無効（固定予算で最後まで学習）"]
    : ["Restart (rollback + lower LR)", "Stop at best",
       "Off (keep training to the budget)"];
  if (modeEl.options.length >= 3) {
    for (let i = 0; i < 3; i += 1) modeEl.options[i].text = optText[i];
  }
  document.querySelectorAll("[data-restart-only]").forEach((el) => {
    el.disabled = mode !== "restart";
  });
  document.querySelectorAll("[data-exp-only]").forEach((el) => {
    el.disabled = mode !== "restart" || !exp;
  });
  const ja = LANG === "ja";
  const text = mode === "restart"
    ? (ja
      ? "再開: 最良チェックポイントへ戻し、学習率を下げて同じデータ順から再試行します。調和モードは 1/2, 1/3, 1/4 …、指数モードは倍率を繰り返し適用します。最大回数0は下限LRまで無制限です。指数倍率は調和モードでは無効です。"
      : "Restart: roll back to the best checkpoint, lower the LR, and retry from the same data order. Harmonic uses 1/2, 1/3, 1/4, ...; exponential repeatedly applies the factor. Max restarts 0 = unlimited until the LR floor. The factor is disabled in harmonic mode.")
    : mode === "stop"
      ? (ja
        ? "停止: 悪化を検出したら最良チェックポイントで停止します。LRは下げず、再開もしません。最良モデルは失われません。"
        : "Stop: stop at the best checkpoint when regression is detected. No LR reduction and no retry; the best model is preserved.")
      : (ja
        ? "無効: 検証ECが悪化しても巻き戻さず、最後まで学習します。最良チェックポイントの保存と best エクスポートは引き続き有効です。"
        : "Off: keep training even when validation CE regresses. Best-checkpoint saving and best export still work.");
  const el = $("guard-explain");
  if (el) el.textContent = text;
}

function saveConfigSoon() {
  clearTimeout(configSaveTimer);
  configSaveTimer = setTimeout(async () => {
    try {
      await api("/api/config", { config: readConfig() });
    } catch (_) {
      /* keep the UI responsive; the explicit Start button still validates */
    }
  }, 650);
}

/* ------------------------------------------------------------------ */
function renderSources(specs) {
  const tb = document.querySelector("#sources-table tbody");
  tb.innerHTML = "";
  specs.forEach((s) => addSourceRow(s));
}

function addSourceRow(s = {}) {
  const tb = document.querySelector("#sources-table tbody");
  const tr = document.createElement("tr");
  const vals = [s.source || "", s.split || "train", s.name || "", s.format || "",
                s.weight ?? 1.0, s.subset ?? ""];
  const tips = [
    TIP("row-source"), TIP("row-split"), TIP("row-name"), TIP("row-format"),
    TIP("row-weight"), TIP("row-subset"),
  ];
  tr.innerHTML =
    vals.map((v, i) =>
      `<td><input autocomplete="off" autocorrect="off" autocapitalize="off" spellcheck="false" ` +
      `data-lpignore="true" data-1p-ignore="true" title="${tips[i]}" ` +
      `style="width:${[260, 70, 90, 80, 60, 70][i]}px" value="${v}"></td>`).join("") +
    `<td><button class="danger">${T("btn_remove")}</button></td>`;
  tr.querySelector("button").onclick = () => tr.remove();
  tb.appendChild(tr);
}

/* every non-empty source row (used by the preview button) */
function allSpecs() {
  const out = [];
  document.querySelectorAll("#sources-table tbody tr").forEach((row) => {
    const g = (c) => row.querySelector(`td:nth-child(${c}) input`).value;
    const source = g(1).trim();
    if (!source) return;
    out.push({
      source, split: g(2).trim() || "train", name: g(3).trim() || null,
      format: g(4).trim() || null, weight: parseFloat(g(5)) || 1.0,
      subset: g(6).trim() ? parseInt(g(6)) : null,
    });
  });
  return out;
}

/* render a normalized example as an input/output table (SFT) or text (PT) */
function renderExampleTable(examples) {
  if (!examples || !examples.length) return `<em>${T("no_data")}</em>`;
  let html = "";
  examples.forEach((ex, i) => {
    html += `<div class="ex-head">例 ${i + 1} / Example ${i + 1}</div>`;
    if (ex.kind === "chat") {
      html += `<table class="io-table"><thead><tr>` +
        `<th>${T("col_role")}</th><th>${T("col_output")}</th></tr></thead><tbody>`;
      let pendingInput = "";
      ex.messages.forEach((m) => {
        const role = T("role_" + m.role) || m.role;
        if (m.role === "assistant") {
          html += `<tr><td class="io-in"><b>${T("col_input")}</b><br>${pendingInput || "-"}` +
                  `</td><td class="io-out"><b>${role}</b><br>${m.content}</td></tr>`;
          pendingInput = "";
        } else {
          pendingInput += (pendingInput ? "<br><br>" : "") +
            `<small>[${role}]</small><br>${m.content}`;
        }
      });
      html += `</tbody></table>`;
    } else {
      html += `<div class="preview">${(ex.text || "").slice(0, 900)}</div>`;
    }
  });
  return html;
}

/* ------------------------------------------------------------------ */
async function previewSource() {
  const specs = allSpecs();
  if (!specs.length) { toast(T("err") + ": source", true); return; }
  $("preview-box").style.display = "block";
  $("preview-out").innerHTML = "<em>...</em>";
  $("preview-table").innerHTML = "";
  try {
    const r = await api("/api/dataset/preview", { specs, n: 3 });
    const sources = r.sources || [];
    if (!sources.length && r.error) throw new Error(r.error);
    const rawBlocks = [];
    const exampleBlocks = [];
    sources.forEach((s, i) => {
      const letter = String.fromCharCode(65 + i);
      if (s.error) {
        rawBlocks.push(`<div class="src-preview error"><b>${letter} ${esc(s.source)}</b><br>${esc(s.error)}</div>`);
        return;
      }
      rawBlocks.push(
        `<div class="src-preview"><b>${letter} ${esc(s.source)}</b>` +
        `<div class="hint">rows=${s.total} | columns=${esc((s.columns || []).join(", "))}</div>` +
        `<pre>${esc((s.rows || []).map((row) => JSON.stringify(row.row, null, 1)).join("\n---\n").slice(0, 1800))}</pre></div>`);
      exampleBlocks.push(`<div class="src-preview"><b>${letter} ${esc(s.source)}</b>` +
        renderExampleTable(s.first_examples) + `</div>`);
    });
    $("preview-out").innerHTML = rawBlocks.join("");
    $("preview-table").innerHTML = exampleBlocks.join("");
  } catch (e) {
    $("preview-out").textContent = T("err") + ": " + e.message;
    $("preview-table").innerHTML = "";
  }
}

/* colour-coded provenance of the first 20 examples/sequences.  The hue comes
   from the legend index, so A/B/C are always visually distinct (v8 hashed the
   source string and could give two sources almost the same colour). */
const SOURCE_HUES = [205, 22, 145, 45, 280, 330, 175, 95, 250, 5, 120, 305];

function sourceBadge(label, source, extra, hue) {
  const text = String(label || "?");
  let h = Number.isFinite(hue) ? hue : 0;
  if (!Number.isFinite(hue)) {
    for (const ch of String(source || text)) h = (h * 31 + ch.charCodeAt(0)) % 360;
  }
  return `<span class="src-badge" style="--h:${h}" title="${esc(source || "")}${extra ? " · " + esc(extra) : ""}">${esc(text)}</span>`;
}

function renderSourceOrder(s) {
  if (!s || (!s.example_order?.length && !s.sequence_sources?.length)) return "";
  const hueByLabel = {};
  (s.source_legend || []).forEach((x, i) => {
    hueByLabel[x.label] = SOURCE_HUES[i % SOURCE_HUES.length];
  });
  const legend = (s.source_legend || []).map((x) =>
    `<span class="legend-item">${sourceBadge(x.label, x.source, "", hueByLabel[x.label])} <span>${esc(x.source)}</span></span>`
  ).join("");
  const ex = (s.example_order || []).map((x) =>
    sourceBadge(x.label, x.source, x.kind, hueByLabel[x.label])
  ).join("");
  const seq = (s.sequence_sources || []).map((x) =>
    sourceBadge(x.label, x.source, (x.labels || []).join("+"), hueByLabel[x.label])
  ).join("");
  return `<div class="order-box">
    <h3>${T("t_order_vis")}</h3>
    <div class="hint">${T("t_order_legend")}: ${legend || "-"}</div>
    <div class="order-row"><b>${LANG === "ja" ? "先頭20例（色分け）" : "First 20 examples (colour)"}</b><div class="order-badges">${ex || "-"}</div></div>
    <div class="order-row"><b>${T("t_order_sequences")}</b><div class="order-badges">${seq || "-"}</div></div>
  </div>`;
}

/* ------------------------------------------------------------------ */
let hashHandled = false;

async function refreshState() {
  STATE = await api("/api/state");
  $("ver").textContent = "v" + STATE.version;
  writeConfig(STATE.config);
  await refreshModels();
  refreshRuns();
  refreshDatasets();
}

function handleHash() {
  const hash = decodeURIComponent(location.hash.replace(/^#/, ""));
  if (hash.startsWith("results:")) {
    const run = hash.slice("results:".length).trim();
    if (run) {
      $("run-select").value = run;
      switchTab("results", false);
      hashHandled = true;
      return;
    }
  }
  const tab = ["model", "data", "train", "results"].includes(hash) ? hash : "model";
  switchTab(tab, false);
}

async function refreshRuns() {
  if (!STATE) return;
  const sel = $("run-select");
  sel.innerHTML = (STATE.runs || []).map((r) => `<option>${r}</option>`).join("");
  if (sel.options.length && !sel.value) sel.value = sel.options[0].value;
}

async function refreshDatasets() {
  if (!STATE) return;
  const el = $("cached-list");
  const datasets = STATE.datasets || [];
  el.innerHTML = datasets.map((d) => {
    const s = d.stats || {};
    let inner = `<div class="chip" title="${esc(d.name)}">${esc(d.name)} — ${d.mb}MB | ` +
      `${s.sequences ?? "?"} seq | ${((s.tokens_total || 0) / 1e6).toFixed(1)}M tok | ` +
      `eps=${s.examples_per_step_setting ?? 0}</div>`;
    if (s.first_examples && s.first_examples.length) {
      inner += `<details class="ds-preview"><summary>${T("t_examples_preview")}</summary>` +
        renderExampleTable(s.first_examples) + `</details>`;
    }
    if (s.example_order?.length || s.sequence_sources?.length) {
      inner += `<details class="ds-preview"><summary>${T("t_order_vis")}</summary>` +
        renderSourceOrder(s) + `</details>`;
    }
    return inner;
  }).join("") || `<span class="hint">-</span>`;

  const sel = $("d-cache");
  if (sel) {
    const wanted = String(sel.dataset.wanted ?? "").replace(/\.pt$/i, "");
    const current = String(sel.value || "").replace(/\.pt$/i, "");
    const keep = current || wanted;
    sel.innerHTML = `<option value="">${LANG === "ja"
      ? "自動（設定から決まる名前のキャッシュ / なければ新規構築）"
      : "Auto (cache name derived from settings; build if missing)"}</option>` +
      datasets.map((d) => {
        const s = d.stats || {};
        return `<option value="${esc(d.name)}">${esc(d.name)} — ${d.mb}MB / ` +
          `${s.sequences ?? "?"} seq</option>`;
      }).join("");
    if (keep && datasets.some((d) => d.name === keep)) sel.value = keep;
    else if (!keep) sel.value = "";
    sel.dataset.wanted = "";
  }
}

/* ------------------------------------------------------------------ */
async function modelInfo() {
  const src = $("model-source").value.trim();
  if (!src) return;
  $("model-info").innerHTML = "<em>...</em>";
  try {
    const info = await api("/api/model/info", { source: src });
    $("model-info").innerHTML = Object.entries(info)
      .filter(([k, v]) => v !== null && k !== "supported")
      .map(([k, v]) => `<div><b>${k}</b><span>${Array.isArray(v) ? v.join(", ").slice(0, 90) : String(v).slice(0, 120)}</span></div>`)
      .join("");
  } catch (e) {
    $("model-info").innerHTML = `<div class="error">${e.message}</div>`;
  }
}

/* ------------------------------------------------------------------ */
let buildTimer = null;
async function startBuild() {
  const cfg = readConfig();
  if (!cfg.dataset_specs.length) {
    toast(T("err") + ": dataset source", true);
    return;
  }
  await api("/api/config", { config: cfg });
  $("build-progress").style.display = "block";
  $("build-progress-wrap").style.display = "block";
  $("build-bar").style.width = "0%";
  $("build-percent").textContent = "0%";
  $("build-status").textContent = T("building");
  try {
    await api("/api/dataset/build", { config: cfg });
  } catch (e) {
    $("build-status").textContent = T("build_error") + ": " + e.message;
    return;
  }
  buildTimer = setInterval(pollBuild, 1200);
}

async function pollBuild() {
  const st = await api("/api/build/status");
  const pct = Math.max(0, Math.min(100, Number(st.percent) || 0));
  $("build-bar").style.width = pct + "%";
  $("build-percent").textContent =
    `${st.phase || ""} ${pct.toFixed(0)}%  ${st.message || ""}`.trim();
  $("build-progress").textContent = (st.progress || []).slice(-8).join("\n");
  if (!st.running) {
    clearInterval(buildTimer);
    if (!st.error) $("build-bar").style.width = "100%";
    if (st.error) {
      $("build-status").textContent = T("build_error") + ": " + st.error;
    } else {
      $("build-status").textContent = T("build_done") + ": " + st.done;
      const s = st.stats || {};
      $("build-stats").innerHTML = Object.entries(s)
        .filter(([k, v]) => typeof v !== "object")
        .map(([k, v]) => `<div><b>${k}</b><span>${v}</span></div>`).join("");
      $("build-order").innerHTML = renderSourceOrder(s);
      $("build-preview").innerHTML =
        `<h3>${T("t_examples_preview")}</h3>` + renderExampleTable(s.first_examples);
      STATE = await api("/api/state");
      if (st.done) $("d-cache").dataset.wanted = String(st.done).replace(/\.pt$/i, "");
      refreshDatasets();
    }
  }
}

/* ------------------------------------------------------------------ */
async function startTraining() {
  const cfg = readConfig();
  if (!cfg.model_source) { toast(T("err") + ": model source", true); return; }
  if (!cfg.dataset_specs.length && !cfg.dataset_cache) {
    toast(T("err") + ": dataset source", true);
    return;
  }
  try {
    await api("/api/config", { config: cfg });
    const r = await api("/api/train/start", { config: cfg, resume: $("resume").checked });
    toast(T("started") + " (pid " + r.pid + ")");
    document.body.innerHTML = `<div style="padding:40px;font-size:18px">` +
      (LANG === "ja"
        ? "学習を開始しました。このページは閉じてかまいません。<br>経過は新しいコンソールウィンドウに表示されます。終了後もログを確認できるようコンソールは開いたままです（Enterで閉じます）。その後このUIが自動で再起動します。"
        : "Training started. You can close this page.<br>Progress is shown in the new console window, which stays open after the run so you can read the log (press Enter to close). This UI then relaunches automatically.") +
      `</div>`;
  } catch (e) {
    toast(T("err") + ": " + e.message, true);
  }
}

/* ------------------------------------------------------------------ */
async function generateCliCommand() {
  const out = $("cli-command");
  try {
    const r = await api("/api/cli-command", {
      config: readConfig(),
      resume: $("resume").checked,
    });
    const command = String(r.command || "");
    out.dataset.command = command;
    out.textContent = command || T("cli_empty");
    $("btn-cli-copy").disabled = !command;
    $("btn-cli-download").disabled = !command;
  } catch (e) {
    out.dataset.command = "";
    out.textContent = T("err") + ": " + e.message;
    $("btn-cli-copy").disabled = true;
    $("btn-cli-download").disabled = true;
  }
}

async function copyCliCommand() {
  const text = $("cli-command").dataset.command || "";
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    toast(T("cli_copied"));
  } catch (_) {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
    toast(T("cli_copied"));
  }
}

function downloadCliCommand() {
  const text = $("cli-command").dataset.command || "";
  if (!text) return;
  const runName = String(readConfig().run_name || "run1")
    .replace(/[^A-Za-z0-9_.-]+/g, "_");
  const header = "# CPU-FineTuning CLI command (PowerShell)\n" +
    "# Run from the project folder containing run.py.\n";
  // PowerShell 5.1 treats UTF-8 without a BOM as ANSI.  Include the BOM so
  // non-ASCII model/data paths survive when the .ps1 is saved and run.
  const blob = new Blob(["\ufeff" + header + text + "\r\n"], {
    type: "text/plain;charset=utf-8",
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `cpu-finetuning-${runName}.ps1`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  toast(T("cli_downloaded"));
}

/* ------------------------------------------------------------------ */
function _mx(v, digits = 4) {
  if (v === null || v === undefined || v === "") return "-";
  if (typeof v === "number" && Number.isFinite(v)) {
    if (Math.abs(v) < 0.001 && v !== 0) return v.toExponential(2);
    return String(Number(v.toFixed(digits)));
  }
  return String(v);
}

function _metricRow(label, value, helpKey) {
  const help = (METRIC_HELP[LANG] || {})[helpKey] || "";
  return `<div class="metric" title="${esc(help)}"><b>${esc(label)}</b>` +
    `<span>${esc(value ?? "-")}</span>` +
    (help ? `<small>${esc(help)}</small>` : "") + `</div>`;
}

function _eventRows(summary, valRows) {
  const ja = LANG === "ja";
  const label = {
    baseline: ja ? "ベースライン" : "baseline",
    val: ja ? "検証" : "validation",
    worsening: ja ? "悪化" : "worsening",
    best: ja ? "最良更新" : "new best",
    regression: ja ? "悪化検出" : "regression",
    restart: ja ? "再開（LR低減）" : "restart (LR reduced)",
    restart_floor: ja ? "LR下限で停止" : "LR floor",
    stop: ja ? "停止" : "stop",
    schedule: ja ? "LRスケジュール補正" : "LR horizon adjusted",
    final_eval: ja ? "最終検証" : "final evaluation",
    export: ja ? "エクスポート" : "export",
    plots: ja ? "PNGグラフ" : "PNG graphs",
  };
  const items = [];
  for (const e of (summary.events || [])) {
    let kind = String(e.kind || "");
    if (kind === "val" && e.meaningful_improvement === false) {
      const bad = (e.best != null && e.val_ce != null
        && Number(e.val_ce) > Number(e.best) + Number(summary.val_min_delta || 0));
      if (!bad) continue;
      kind = "worsening";
    }
    if (kind === "plots" && e.status !== "done") continue;
    let detail = "";
    if (e.step !== undefined) detail += `step ${e.step}`;
    if (e.val_ce !== undefined && e.val_ce !== null) detail += ` · CE ${_mx(e.val_ce)}`;
    if (e.best !== undefined && e.best !== null) detail += ` · best ${_mx(e.best)}`;
    if (e.best_step !== undefined) detail += ` @ ${e.best_step}`;
    if (e.retry !== undefined) detail += ` · retry ${e.retry}`;
    if (e.lr !== undefined) detail += ` · LR ${Number(e.lr).toExponential(2)}`;
    if (e.old_total !== undefined) detail += ` · horizon ${e.old_total}→${e.new_total}`;
    if (e.sec_per_step !== undefined) detail += ` · ${e.sec_per_step}s/step`;
    if (e.path) detail += ` · ${e.path}`;
    if (e.error) detail += ` · ${e.error}`;
    items.push(`<li><b>${esc(label[kind] || kind)}</b> <span>${esc(detail)}</span></li>`);
  }
  // Older v8 summaries have no events array: derive the useful milestones.
  if (!(summary.events || []).length) {
    const seenBest = new Set();
    for (const h of (valRows || [])) {
      if (h.step === 0) {
        items.push(`<li><b>${label.baseline}</b> <span>step 0 · CE ${_mx(h.val_ce)}</span></li>`);
      }
      if (h.step === summary.val_ce_best_step && !seenBest.has(h.step)) {
        seenBest.add(h.step);
        items.push(`<li><b>${label.best}</b> <span>step ${h.step} · CE ${_mx(h.val_ce)}</span></li>`);
      } else if (h.best_ce !== undefined && h.val_ce > h.best_ce + (summary.val_min_delta || 0)) {
        items.push(`<li><b>${label.regression}</b> <span>step ${h.step} · CE ${_mx(h.val_ce)} &gt; best ${_mx(h.best_ce)}</span></li>`);
      }
    }
    for (const rr of (summary.restarts || [])) {
      items.push(`<li><b>${label.restart}</b> <span>from step ${rr.from_step} · retry ${rr.retry} · LR ${Number(rr.lr).toExponential(2)}</span></li>`);
    }
  }
  if (!items.length) return "";
  return `<div class="event-box"><h3>${T("t_events")}</h3><ul class="event-list">${items.join("")}</ul></div>`;
}

function _plotBox(run, plots) {
  if (!plots || !Object.keys(plots).length) return "";
  const names = {
    lr: T("plot_lr"), train_loss: T("plot_loss"), grad_norm: T("plot_grad"),
    val_ce: T("plot_val"), training_overview: LANG === "ja" ? "全体概要" : "Overview",
    overview: LANG === "ja" ? "全体概要" : "Overview",
  };
  return `<div class="plot-box"><h3>${T("t_plots")}</h3><div class="plot-grid">` +
    Object.entries(plots).map(([key, file]) => {
      const base = String(file).split(/[\\/]/).pop();
      const url = `/api/plot?run=${encodeURIComponent(run)}&name=${encodeURIComponent(base)}`;
      return `<a class="plot-link" href="${url}" target="_blank" rel="noopener">` +
        `<img src="${url}" alt="${esc(names[key] || key)}" loading="lazy">` +
        `<span>${esc(names[key] || key)} · ${esc(base)}</span></a>`;
    }).join("") + `</div></div>`;
}

async function loadResults(run) {
  if (!run) return;
  let r;
  try {
    r = await api("/api/results?run=" + encodeURIComponent(run));
  } catch (e) {
    $("result-summary").innerHTML = `<div class="error">${esc(e.message)}</div>`;
    return;
  }
  const s = r.summary || {};
  const ja = LANG === "ja";
  const stopLabels = {
    time_limit: ja ? "時間制限" : "time limit",
    epoch_limit: ja ? "エポック上限" : "epoch limit",
    step_limit: ja ? "ステップ上限" : "step limit",
    val_regression: ja ? "検証悪化" : "validation regression",
    restart_lr_floor: ja ? "再開LR下限" : "restart LR floor",
    one_epoch_default: ja ? "既定の1エポック" : "one-epoch default",
    completed: ja ? "完了" : "completed",
  };
  const restarts = s.restarts || [];
  const restartLrs = restarts.slice(-8)
    .map((x) => Number(x.lr).toExponential(2)).join(" → ") || "-";
  const rows = [
    [ja ? "実行名" : "Run", s.run_name, "history"],
    [ja ? "モデル" : "Model", s.model, "device"],
    [ja ? "デバイス / 精度" : "Device / dtype", `${s.device || "-"} / ${s.compute_dtype || "-"}`, "device"],
    [ja ? "学習ステップ" : "Run steps", _mx(s.run_steps ?? s.steps), "run_steps"],
    [ja ? "計画ステップ / LR終端" : "Planned / LR horizon", `${_mx(s.planned_steps)} / ${_mx(s.schedule_total)}`, "schedule_total"],
    [ja ? "終了理由" : "Stop reason", stopLabels[s.stop_reason] || s.stop_reason || "-", "stop_reason"],
    [ja ? "損失 first→last" : "Loss first→last", `${_mx(s.loss_first)} → ${_mx(s.loss_last)} (k=${_mx(s.loss_window)})`, "loss"],
    [ja ? "損失 EMA first→last" : "Loss EMA first→last", `${_mx(s.loss_ema_first)} → ${_mx(s.loss_ema_last)}`, "loss_ema"],
    [ja ? "損失 最小" : "Loss min", _mx(s.loss_min), "loss_min"],
    [ja ? "検証CE baseline→last" : "Val CE baseline→last", `${_mx(s.val_ce_first)} → ${_mx(s.val_ce_last ?? s.val_ce_final)}`, "val_baseline"],
    [ja ? "検証CE best" : "Val CE best", `${_mx(s.val_ce_best)} @ step ${_mx(s.val_ce_best_step)}`, "val_best"],
    [ja ? "検証CE final / 改善量" : "Val CE final / delta", `${_mx(s.val_ce_final)} / ${_mx(s.val_delta_base)}`, "val_delta"],
    [ja ? "学習プローブ first→last" : "Train probe first→last", `${_mx(s.train_probe_first)} → ${_mx(s.train_probe_last)}`, "probe"],
    [ja ? "勾配ノルム mean / max" : "Grad norm mean / max", `${_mx(s.grad_norm_mean, 5)} / ${_mx(s.grad_norm_max, 5)}`, "grad_mean"],
    [ja ? "勾配スパイク" : "Gradient spikes", _mx(s.grad_spikes ?? 0), "spikes"],
    [ja ? "秒/step / step/時" : "sec/step / steps/hour", `${_mx(s.sec_per_step, 3)} / ${_mx(s.steps_per_hour, 1)}`, "sec_step"],
    [ja ? "トークン/秒" : "Tokens/sec", _mx(s.tokens_per_second, 1), "tokens_sec"],
    [ja ? "ガード" : "Guard", `${s.guard_mode || "-"} / patience ${_mx(s.val_patience)} / minΔ ${_mx(s.val_min_delta, 5)}`, "guard"],
    [ja ? "再開回数 / LR" : "Restarts / LR", `${restarts.length} / ${restartLrs}`, "restarts"],
    [ja ? "再開ポリシー" : "Restart policy", `${s.restart_lr_mode || "-"} / floor ${s.restart_lr_floor ?? "-"}`, "restart_policy"],
    [ja ? "最良チェックポイント" : "Best checkpoint", s.best_checkpoint || "-", "best_checkpoint"],
    [ja ? "エクスポート元" : "Exported from", `step ${s.exported_step ?? "-"} · CE ${_mx(s.exported_val_ce)}`, "exported"],
    [ja ? "エクスポート先" : "Export path", s.export_path || "-", "export_path"],
    [ja ? "使用キャッシュ" : "Dataset cache", s.dataset_cache || s.dataset?.cache || "-", "cache"],
  ];
  $("result-summary").innerHTML = rows.map(([label, value, key]) =>
    _metricRow(label, value, key)).join("");
  $("result-events").innerHTML = _eventRows(s, r.val || []);
  $("result-plots").innerHTML = _plotBox(run, s.plots);
  const orderHost = $("result-order");
  if (orderHost) {
    orderHost.innerHTML = s.dataset
      ? `<h3>${ja ? "使用データの並び" : "Dataset order used"}</h3>` +
        renderSourceOrder(s.dataset)
      : "";
  }
  const hist = (r.history || []);
  drawChart($("chart-loss"), hist.map((h) => [h.step, h.loss]), T("plot_loss"));
  drawChart($("chart-val"), (r.val || []).map((h) => [h.step, h.val_ce]), T("plot_val"), true);
  drawChart($("chart-lr"), hist.filter((h) => h.lr != null)
    .map((h) => [h.step, h.lr]), T("plot_lr"));
  drawChart($("chart-grad"), hist.filter((h) => h.grad_norm != null)
    .map((h) => [h.step, h.grad_norm]), T("plot_grad"));
  const logEl = $("result-log");
  if (logEl) {
    // Requirement 9: the training output must still be visible after the
    // run ends.  v8 closed the console and left no discoverable log; v9
    // launches the results view and opens this panel by default.
    const logBox = $("result-log-box");
    if (logBox) logBox.open = true;
    logEl.textContent = "...";
    try {
      const log = await api("/api/log?run=" + encodeURIComponent(run));
      logEl.textContent = log.text || (ja ? "（ログなし）" : "(no log)");
      logEl.scrollTop = logEl.scrollHeight;
    } catch (e) {
      logEl.textContent = T("err") + ": " + e.message;
    }
  }
}

function drawChart(cv, pts, label, few) {
  if (!cv) return;
  const ctx = cv.getContext("2d");
  pts = (pts || []).filter((p) => Array.isArray(p)
    && Number.isFinite(Number(p[0])) && Number.isFinite(Number(p[1])));
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.strokeStyle = "#334";
  ctx.strokeRect(0.5, 0.5, cv.width - 1, cv.height - 1);
  if (!pts.length) {
    ctx.fillStyle = "#667";
    ctx.fillText(T("no_data"), 10, 20);
    return;
  }
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs) || 1;
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const px = (x) => 10 + (x - xmin) / Math.max(1, xmax - xmin) * (cv.width - 30);
  const py = (y) => cv.height - 15 - (y - ymin) / Math.max(1e-9, ymax - ymin) * (cv.height - 30);
  ctx.strokeStyle = "#345";
  ctx.beginPath();
  pts.forEach((p, i) => (i ? ctx.lineTo(px(p[0]), py(p[1])) : ctx.moveTo(px(p[0]), py(p[1]))));
  ctx.stroke();
  if (!few && pts.length > 20) {
    ctx.strokeStyle = "rgba(200,60,60,0.9)";
    ctx.beginPath();
    let e = pts[0][1];
    pts.forEach((p, i) => {
      e = 0.98 * e + 0.02 * p[1];
      const x = px(p[0]), y = py(e);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  }
  ctx.fillStyle = "#223";
  const fmt = (v) => (Math.max(Math.abs(ymin), Math.abs(ymax)) < 0.01
    ? Number(v).toExponential(2) : v.toFixed(3));
  ctx.fillText(`${label}  [${fmt(ymin)} … ${fmt(ymax)}]`, 12, 14);
}

/* ------------------------------------------------------------------ */
let MODELS = [];

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

async function refreshModels() {
  try {
    const r = await api("/api/models");
    MODELS = r.models || [];
  } catch (e) {
    MODELS = [];
  }
  const sel = $("gen-model-suggest");
  if (!sel) return;
  const cur = $("gen-model").value.trim();
  const opts = [`<option value="">${esc(T("gen_select"))}</option>`];
  for (const m of MODELS) {
    const name = m.run_name || m.path.split(/[\\/]/).pop();
    const kind = m.kind === "trained" ? T("gen_trained") : T("gen_base");
    const mode = m.last_mode ? ` · ${m.last_mode}` : "";
    opts.push(`<option value="${esc(m.path)}">${esc(kind)}: ${esc(name)}${esc(mode)}</option>`);
  }
  sel.innerHTML = opts.join("");
  if (cur) sel.value = cur;
  if (!sel.value && MODELS.length) {
    sel.value = MODELS[0].path;
  }
}

async function generate() {
  const model = $("gen-model").value.trim();
  if (!model) { toast(T("err") + ": model", true); return; }
  $("gen-out").textContent = "...";
  $("gen-out").style.display = "block";
  $("gen-compare-out").style.display = "none";
  $("btn-gen").disabled = true;
  try {
    const m = MODELS.find((x) => x.path === model);
    const r = await api("/api/generate", {
      model,
      base_model: m ? m.base_model : "",
      prompt: $("gen-prompt").value,
      max_new_tokens: 128,
      device: $("model-device").value || "auto",
      compare: $("gen-compare").checked,
    });
    if (r.compare) {
      $("gen-out").style.display = "none";
      $("gen-compare-out").style.display = "grid";
      $("gen-base-out").textContent = r.base.text || "(empty)";
      $("gen-tuned-out").textContent = r.tuned.text || "(empty)";
      $("t-gen-base").textContent = `${T("gen_base")} (${r.base.model})`;
      $("t-gen-tuned").textContent = `${T("gen_trained")} (${r.tuned.model})`;
    } else {
      $("gen-out").textContent = r.text || "(empty)";
      $("gen-compare-out").style.display = "none";
    }
    await refreshModels();
  } catch (e) {
    $("gen-out").textContent = T("err") + ": " + e.message;
  } finally {
    $("btn-gen").disabled = false;
  }
}

/* ------------------------------------------------------------------ */
function fmtDur(sec) {
  sec = Math.max(0, Number(sec) || 0);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.round(sec % 60);
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s}s`;
  return `${s}s`;
}

async function estimateTime() {
  $("estimate-out").innerHTML = "<em>...</em>";
  try {
    const r = await api("/api/estimate", { config: readConfig() });
    const L = LANG === "ja" ? {
      params: "パラメータ数", examples: "データ件数", seqs: "学習シーケンス",
      steps: "学習ステップ", step: "1ステップ", build: "データ構築",
      train: "学習時間", total: "合計（概算）", note: "注記",
    } : {
      params: "Parameters", examples: "Examples", seqs: "Train sequences",
      steps: "Train steps", step: "Per step", build: "Dataset build",
      train: "Training", total: "Total (approx.)", note: "Note",
    };
    const rows = [
      [L.params, r.params ? (r.params / 1e9).toFixed(2) + "B" : "-"],
      [L.examples, r.examples ?? "-"],
      [L.seqs, r.sequences ?? "-"],
      [L.steps, r.steps ?? "-"],
      [L.step, r.sec_per_step ? r.sec_per_step.toFixed(1) + "s" : "-"],
      [L.build, fmtDur(r.build_seconds)],
      [L.train, fmtDur(r.train_seconds)],
      [L.total, fmtDur(r.total_seconds)],
    ];
    $("estimate-out").innerHTML = rows.map(([k, v]) =>
      `<div><b>${k}</b><span>${v}</span></div>`).join("");
    if (r.note) {
      $("estimate-out").innerHTML +=
        `<div style="grid-column:1/-1"><b>${L.note}</b><span>${esc(r.note)}</span></div>`;
    }
  } catch (e) {
    $("estimate-out").innerHTML = `<div class="error">${esc(e.message)}</div>`;
  }
}

/* ------------------------------------------------------------------ */
function switchTab(name, updateHash = true) {
  document.querySelectorAll("#tabs button").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((s) =>
    s.classList.toggle("active", s.id === "tab-" + name));
  if (updateHash) {
    const target = "#" + name;
    if (location.hash !== target) {
      history.replaceState(null, "", target);
    }
  }
  // v9: loading results must never call refreshState() recursively.  v8's
  // switchTab('results') -> refreshState() -> #results:run -> switchTab loop
  // locked every tab after training.
  if (name === "results") {
    const run = $("run-select").value;
    if (run) loadResults(run);
  }
}

function presetChange() {
  const p = $("t-preset").value;
  if (p === "custom" || !STATE) return;
  const preset = STATE.presets[p];
  if (!preset) return;
  $("a-span").value = preset.ascot.span_length;
  $("a-spans").value = preset.ascot.num_spans;
  $("a-prefix").value = preset.ascot.prefix_len;
  $("a-plan").value = preset.ascot.long_plan_tokens ?? 64;
  $("a-sink").value = preset.ascot.sink_len;
  $("a-slice").value = preset.ascot.slice_size;
  $("a-mode").value = preset.ascot.mode;
  $("a-head").value = preset.ascot.head_positions;
  $("a-headlr").value = preset.ascot.head_lr_scale ?? 0.10;
  $("a-sample").value = preset.ascot.slice_sampling;
  $("a-refresh").value = preset.ascot.shadow_refresh_every;
  $("t-lr-sci").value = fmtSci(preset.lr);
  $("t-lr-dec").value = fmtDec(preset.lr);
}

function ctxSelChange() {
  const v = $("d-ctx-sel").value;
  if (v === "custom") {
    $("d-ctx").style.display = "block";
  } else {
    $("d-ctx").style.display = "none";
    $("d-ctx").value = v;
  }
}

/* source-row tooltips */
Object.assign(TIPS.ja, {
  "row-source": "HuggingFaceのデータセットID、またはローカルファイル（json/jsonl/csv/tsv/parquet/arrow/txt/md）のパス。\" \"で囲まれたパスも自動処理されます。",
  "row-split": "使用する分割（train / validation など）。",
  "row-name": "データセットのconfig名（例: Math）。空欄なら全configを自動結合します（1つだけならそれを使用）。",
  "row-format": "形式を明示指定（空欄で自動判別）: messages / conversations / alpaca / prompt / text など。",
  "row-weight": "複数ソースを混合するときの比率。",
  "row-subset": "先頭N件だけ使う（大きなデータセットの試験に便利）。",
});
Object.assign(TIPS.en, {
  "row-source": "HuggingFace dataset id or a local file path (json/jsonl/csv/tsv/parquet/arrow/txt/md). Quoted paths are unwrapped automatically.",
  "row-split": "Which split to use (train / validation ...).",
  "row-name": "Dataset config name (e.g. Math). Leave blank to combine all configs automatically (or use the only one).",
  "row-format": "Force the format (empty = auto): messages / conversations / alpaca / prompt / text ...",
  "row-weight": "Mixing weight when several sources are combined.",
  "row-subset": "Use only the first N examples.",
});

/* ------------------------------------------------------------------ */
window.addEventListener("load", async () => {
  applyI18n();
  bindLr();
  // Requirement 8: never let the browser remember model paths or prompts.
  document.querySelectorAll("input,textarea").forEach((el) => {
    el.autocomplete = "off";
    el.setAttribute("autocorrect", "off");
    el.setAttribute("autocapitalize", "off");
    el.spellcheck = false;
    el.setAttribute("data-lpignore", "true");
    el.setAttribute("data-1p-ignore", "true");
  });
  try { await refreshState(); } catch (e) { toast(e.message, true); }
  handleHash();
  document.querySelectorAll("#tabs button").forEach((b) =>
    b.addEventListener("click", () => switchTab(b.dataset.tab)));
  window.addEventListener("hashchange", handleHash);
  const setLang = (lang) => {
    LANG = lang;
    applyI18n();
    refreshDatasets();
    if (document.querySelector("#tab-results.active")) {
      const run = $("run-select").value;
      if (run) loadResults(run);
    }
  };
  $("lang-ja").onclick = () => setLang("ja");
  $("lang-en").onclick = () => setLang("en");
  $("btn-model-info").onclick = modelInfo;
  $("btn-add-source").onclick = () => addSourceRow();
  $("btn-preview").onclick = previewSource;
  $("btn-build").onclick = startBuild;
  $("btn-start").onclick = startTraining;
  $("btn-cli-export").onclick = generateCliCommand;
  $("btn-cli-copy").onclick = copyCliCommand;
  $("btn-cli-download").onclick = downloadCliCommand;
  $("btn-refresh").onclick = () => loadResults($("run-select").value);
  $("run-select").onchange = (e) => {
    const run = e.target.value;
    if (run) {
      history.replaceState(null, "", "#results:" + encodeURIComponent(run));
      loadResults(run);
    }
  };
  $("btn-gen").onclick = generate;
  $("gen-model-suggest").onchange = (e) => {
    if (e.target.value) $("gen-model").value = e.target.value;
  };
  $("btn-estimate").onclick = estimateTime;
  $("t-preset").onchange = presetChange;
  $("d-ctx-sel").onchange = ctxSelChange;
  $("model-source").addEventListener("change", modelInfo);
  $("model-source").addEventListener("change", saveConfigSoon);
  $("model-device").addEventListener("change", saveConfigSoon);
  $("model-dtype").addEventListener("change", saveConfigSoon);
  $("btn-cache-refresh").onclick = async () => {
    try { await refreshState(); } catch (e) { toast(e.message, true); }
  };
  $("d-cache").addEventListener("change", saveConfigSoon);
  $("t-guardmode").addEventListener("change", () => {
    applyGuardUi();
    saveConfigSoon();
  });
  $("t-restartmode").addEventListener("change", () => {
    applyGuardUi();
    saveConfigSoon();
  });
  ["t-patience", "t-mindelta", "t-exportbest", "t-restartdecay",
   "t-restartfloor", "t-maxrestarts", "t-probe", "t-maxhours",
   "t-maxepochs", "t-steps", "t-lr-sci", "t-lr-dec"].forEach((id) => {
    const el = $(id);
    if (el) el.addEventListener("change", saveConfigSoon);
  });
});
