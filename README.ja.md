# YASAKANI SYCL Qwen3.5

[English](README.md) · [Windowsビルドとモデル変換](docs/WINDOWS-QWEN35.md) · [ベンチマーク詳細](docs/QWEN35-BENCHMARKS.md)

[YASAKANI v0.1.6](https://github.com/yasakani-dev/yasakani/tree/v0.1.6) をベースに、Qwen3.5のテキスト推論を追加した実験版です。WindowsネイティブのSYCL DLLは、SYCLのNVIDIA CUDAバックエンドを使います。`.yskn` を読み込み、prefillの一括処理と1トークンずつの生成に対応します。

| モデル | バックエンド | 実機検証 |
|---|---|---|
| Qwen3.5-4B | SYCL・CUDA | Windows、RTX 5060 Ti・RTX 3060、speed・quality |
| Qwen3.5-9B | SYCL・CUDA | Windows、RTX 5060 Ti、speed・quality |

DLLはモデルサイズとGPUアーキテクチャを指定してビルドします。9BのSYCL版はRTX 3060向けの `sm_86` ビルドも確認済みです。実機での推論確認は表のとおりです。

RTX 5060 Tiの長文読解では、SYCL版の生成速度は4Bで **161.54 / 140.44 tok/s**、9Bで **102.89 / 85.26 tok/s** でした（speed / quality）。RTX 3060での4B実測は **114.50 / 104.18 tok/s** です。CUDA版との比較、TTFT、VRAM使用量、正解数は[ベンチマーク詳細](docs/QWEN35-BENCHMARKS.md)にまとめています。

リポジトリにはソース、ベンチマーク入力、測定記録を収録しています。[変換スクリプト](tools/convert/convert_qwen35.py)で4B・9Bそれぞれのspeed／qualityパックを作れます。同じパックをSYCL版とCUDA版で使えます。ビルド、モデル変換、C APIの使い方は[手順書](docs/WINDOWS-QWEN35.md)を参照してください。

上流の[Apache-2.0ライセンス](LICENSE)、[NOTICE](NOTICE)、[第三者の帰属表示](THIRD_PARTY_NOTICES.md)を引き継いでいます。元モデルのライセンスはQwenの各配布元を参照してください。
