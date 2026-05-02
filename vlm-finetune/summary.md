PHASE    SCRIPT                      OUTPUT                          NOTES
─────────────────────────────────────────────────────────────────────────────
1.1      generate_cnn_outputs.py     cnn_outputs.json                ~10,000 records
                                     gradcam/*.npy + *_overlay.jpg

1.2      teacher_generation.py       teacher_outputs.json            ~2,500 with GPT-4o
                                                                     Cost: ~\$75-150

2        quality_control.py          clean_dataset.json              ~1,750-2,000 clean
                                     qc_report.json                  Discard rate: 20-30%

3        format_dataset.py           train.jsonl                     ~1,488 train
                                     validation.jsonl                ~175 val
                                     test.jsonl                      ~87 test

4        train_sft.py                checkpoints/sft/final/          ~2-4 hours on A100
                                                                     ~6-10 hours on RTX 4090

5.1      generate_dpo_pairs.py       dpo_automated_pairs.jsonl       350 automated
                                     dpo_manual_template.json        150 manual template

5.2      train_dpo.py                checkpoints/dpo/final/          ~1-2 hours on A100

6        evaluate.py                 evaluation_report.json          QC pass rate target: >85%

7        app.py (updated)            Live web application            CNN + VLM integrated
─────────────────────────────────────────────────────────────────────────────

HARDWARE REQUIREMENTS:

- Training: 1x GPU with 24GB+ VRAM (RTX 4090, A100, or cloud equivalent)
- Inference: 1x GPU with 16GB+ VRAM (RTX 4080+ or cloud)
- Teacher generation: OpenAI API access (GPT-4o)

TOTAL ESTIMATED COST:

- GPT-4o teacher generation: ~\$75-150
- Cloud GPU (if needed): ~\$20-50 for training
- Total: ~\$100-200
