// Sinh tự động bởi build_site.py — không sửa tay
window.T1DATA = {
 "meta": {
  "model": "ViT-B-32",
  "pretrained": "laion2b_s34b_b79k",
  "dim": 512,
  "n_images": 19264,
  "n_images_ok": 19264,
  "n_captions": 96381,
  "split": {
   "train": 15000,
   "val": 2126,
   "test": 2138
  },
  "pool": {
   "train": 46944,
   "val": 2126,
   "test": 2138
  },
  "seed_data": 0,
  "seeds": [
   0,
   1,
   2
  ],
  "gallery_size": 1000,
  "top_k": 100,
  "created": "2026-11-01 04:17:57",
  "source": "Karpathy COCO split (train+restval->train, val->val, test->test) ∩ Visual Genome (coco_id)",
  "alpha": 0.8,
  "bootstrap": {
   "MRR": {
    "delta": 1.0640656492287144,
    "ci95": [
     0.7401275109371445,
     1.373771472958955
    ],
    "p_value": 0.0
   },
   "R@1": {
    "delta": 0.8694839192221391,
    "ci95": [
     0.34592370979805537,
     1.3836948391922215
    ],
    "p_value": 0.003
   },
   "R@10": {
    "delta": 1.206058339566193,
    "ci95": [
     0.8414360508601346,
     1.5706806282722514
    ],
    "p_value": 0.0
   }
  }
 },
 "metrics": [
  {
   "method": "clip",
   "direction": "t2i",
   "split": "test",
   "protocol": "seed0",
   "gallery": 1000,
   "n": 5001,
   "MedR": 1.0,
   "R@1": 58.17,
   "R@5": 84.28,
   "R@10": 91.46,
   "MRR": 69.79,
   "nDCG@10": 74.67
  },
  {
   "method": "clip",
   "direction": "t2i",
   "split": "test",
   "protocol": "seed1",
   "gallery": 1000,
   "n": 5002,
   "MedR": 1.0,
   "R@1": 56.02,
   "R@5": 83.75,
   "R@10": 91.58,
   "MRR": 68.27,
   "nDCG@10": 73.54
  },
  {
   "method": "clip",
   "direction": "t2i",
   "split": "test",
   "protocol": "seed2",
   "gallery": 1000,
   "n": 5001,
   "MedR": 1.0,
   "R@1": 57.89,
   "R@5": 84.2,
   "R@10": 92.22,
   "MRR": 69.62,
   "nDCG@10": 74.74
  },
  {
   "method": "clip",
   "direction": "t2i",
   "split": "test",
   "protocol": "full",
   "gallery": 2138,
   "n": 10696,
   "MedR": 2.0,
   "R@1": 48.31,
   "R@5": 75.35,
   "R@10": 84.08,
   "MRR": 60.4,
   "nDCG@10": 65.51
  },
  {
   "method": "clip",
   "direction": "t2i",
   "split": "test",
   "protocol": "mean3",
   "gallery": 1000,
   "n": 5001,
   "MedR": null,
   "R@1": 57.36,
   "R@5": 84.08,
   "R@10": 91.76,
   "MRR": 69.23,
   "nDCG@10": 74.32
  },
  {
   "method": "clip",
   "direction": "t2i",
   "split": "test",
   "protocol": "std3",
   "gallery": 1000,
   "n": 5001,
   "MedR": null,
   "R@1": 0.95,
   "R@5": 0.24,
   "R@10": 0.33,
   "MRR": 0.68,
   "nDCG@10": 0.55
  },
  {
   "method": "clip",
   "direction": "t2i",
   "split": "val",
   "protocol": "seed0",
   "gallery": 1000,
   "n": 5001,
   "MedR": 1.0,
   "R@1": 58.73,
   "R@5": 85.58,
   "R@10": 92.7,
   "MRR": 70.48,
   "nDCG@10": 75.55
  },
  {
   "method": "clip",
   "direction": "t2i",
   "split": "val",
   "protocol": "seed1",
   "gallery": 1000,
   "n": 5003,
   "MedR": 1.0,
   "R@1": 57.85,
   "R@5": 84.71,
   "R@10": 92.44,
   "MRR": 69.76,
   "nDCG@10": 74.92
  },
  {
   "method": "clip",
   "direction": "t2i",
   "split": "val",
   "protocol": "seed2",
   "gallery": 1000,
   "n": 5001,
   "MedR": 1.0,
   "R@1": 57.93,
   "R@5": 83.92,
   "R@10": 91.72,
   "MRR": 69.53,
   "nDCG@10": 74.52
  },
  {
   "method": "clip",
   "direction": "t2i",
   "split": "val",
   "protocol": "full",
   "gallery": 2126,
   "n": 10633,
   "MedR": 2.0,
   "R@1": 48.55,
   "R@5": 75.39,
   "R@10": 84.36,
   "MRR": 60.61,
   "nDCG@10": 65.73
  },
  {
   "method": "clip",
   "direction": "t2i",
   "split": "val",
   "protocol": "mean3",
   "gallery": 1000,
   "n": 5001,
   "MedR": null,
   "R@1": 58.17,
   "R@5": 84.74,
   "R@10": 92.29,
   "MRR": 69.92,
   "nDCG@10": 75.0
  },
  {
   "method": "clip",
   "direction": "t2i",
   "split": "val",
   "protocol": "std3",
   "gallery": 1000,
   "n": 5001,
   "MedR": null,
   "R@1": 0.4,
   "R@5": 0.68,
   "R@10": 0.41,
   "MRR": 0.4,
   "nDCG@10": 0.42
  },
  {
   "method": "clip",
   "direction": "i2t",
   "split": "test",
   "protocol": "seed0",
   "gallery": 1000,
   "n": 1000,
   "MedR": 1.0,
   "R@1": 74.4,
   "R@5": 91.8,
   "R@10": 96.5,
   "MRR": 82.29,
   "nDCG@10": 85.59
  },
  {
   "method": "clip",
   "direction": "i2t",
   "split": "test",
   "protocol": "seed1",
   "gallery": 1000,
   "n": 1000,
   "MedR": 1.0,
   "R@1": 72.8,
   "R@5": 91.3,
   "R@10": 96.3,
   "MRR": 80.64,
   "nDCG@10": 84.26
  },
  {
   "method": "clip",
   "direction": "i2t",
   "split": "test",
   "protocol": "seed2",
   "gallery": 1000,
   "n": 1000,
   "MedR": 1.0,
   "R@1": 75.1,
   "R@5": 92.6,
   "R@10": 96.8,
   "MRR": 82.54,
   "nDCG@10": 85.86
  },
  {
   "method": "clip",
   "direction": "i2t",
   "split": "test",
   "protocol": "full",
   "gallery": 2138,
   "n": 2138,
   "MedR": 1.0,
   "R@1": 65.53,
   "R@5": 86.01,
   "R@10": 91.44,
   "MRR": 74.65,
   "nDCG@10": 78.38
  },
  {
   "method": "clip",
   "direction": "i2t",
   "split": "test",
   "protocol": "mean3",
   "gallery": 1000,
   "n": 1000,
   "MedR": null,
   "R@1": 74.1,
   "R@5": 91.9,
   "R@10": 96.53,
   "MRR": 81.82,
   "nDCG@10": 85.24
  },
  {
   "method": "clip",
   "direction": "i2t",
   "split": "test",
   "protocol": "std3",
   "gallery": 1000,
   "n": 1000,
   "MedR": null,
   "R@1": 0.96,
   "R@5": 0.54,
   "R@10": 0.21,
   "MRR": 0.84,
   "nDCG@10": 0.7
  },
  {
   "method": "clip",
   "direction": "i2t",
   "split": "val",
   "protocol": "seed0",
   "gallery": 1000,
   "n": 1000,
   "MedR": 1.0,
   "R@1": 74.8,
   "R@5": 93.3,
   "R@10": 97.6,
   "MRR": 82.79,
   "nDCG@10": 86.29
  },
  {
   "method": "clip",
   "direction": "i2t",
   "split": "val",
   "protocol": "seed1",
   "gallery": 1000,
   "n": 1000,
   "MedR": 1.0,
   "R@1": 75.3,
   "R@5": 93.3,
   "R@10": 97.8,
   "MRR": 83.16,
   "nDCG@10": 86.61
  },
  {
   "method": "clip",
   "direction": "i2t",
   "split": "val",
   "protocol": "seed2",
   "gallery": 1000,
   "n": 1000,
   "MedR": 1.0,
   "R@1": 71.1,
   "R@5": 92.3,
   "R@10": 96.8,
   "MRR": 80.13,
   "nDCG@10": 84.07
  },
  {
   "method": "clip",
   "direction": "i2t",
   "split": "val",
   "protocol": "full",
   "gallery": 2126,
   "n": 2126,
   "MedR": 1.0,
   "R@1": 65.19,
   "R@5": 87.3,
   "R@10": 92.99,
   "MRR": 74.95,
   "nDCG@10": 79.03
  },
  {
   "method": "clip",
   "direction": "i2t",
   "split": "val",
   "protocol": "mean3",
   "gallery": 1000,
   "n": 1000,
   "MedR": null,
   "R@1": 73.73,
   "R@5": 92.97,
   "R@10": 97.4,
   "MRR": 82.03,
   "nDCG@10": 85.65
  },
  {
   "method": "clip",
   "direction": "i2t",
   "split": "val",
   "protocol": "std3",
   "gallery": 1000,
   "n": 1000,
   "MedR": null,
   "R@1": 1.87,
   "R@5": 0.47,
   "R@10": 0.43,
   "MRR": 1.35,
   "nDCG@10": 1.13
  },
  {
   "method": "clip_lex",
   "direction": "t2i",
   "split": "test",
   "protocol": "seed0",
   "gallery": 1000,
   "n": 5001,
   "MedR": 1.0,
   "R@1": 59.87,
   "R@5": 85.5,
   "R@10": 92.12,
   "MRR": 71.1,
   "nDCG@10": 75.85
  },
  {
   "method": "clip_lex",
   "direction": "t2i",
   "split": "test",
   "protocol": "seed1",
   "gallery": 1000,
   "n": 5002,
   "MedR": 1.0,
   "R@1": 56.64,
   "R@5": 84.39,
   "R@10": 92.44,
   "MRR": 69.09,
   "nDCG@10": 74.4
  },
  {
   "method": "clip_lex",
   "direction": "t2i",
   "split": "test",
   "protocol": "seed2",
   "gallery": 1000,
   "n": 5001,
   "MedR": 1.0,
   "R@1": 58.87,
   "R@5": 84.88,
   "R@10": 92.68,
   "MRR": 70.43,
   "nDCG@10": 75.47
  },
  {
   "method": "clip_lex",
   "direction": "t2i",
   "split": "test",
   "protocol": "full",
   "gallery": 2138,
   "n": 10696,
   "MedR": 2.0,
   "R@1": 49.18,
   "R@5": 76.34,
   "R@10": 85.28,
   "MRR": 61.46,
   "nDCG@10": 66.65
  },
  {
   "method": "clip_lex",
   "direction": "t2i",
   "split": "test",
   "protocol": "mean3",
   "gallery": 1000,
   "n": 5001,
   "MedR": null,
   "R@1": 58.46,
   "R@5": 84.92,
   "R@10": 92.42,
   "MRR": 70.21,
   "nDCG@10": 75.24
  },
  {
   "method": "clip_lex",
   "direction": "t2i",
   "split": "test",
   "protocol": "std3",
   "gallery": 1000,
   "n": 5001,
   "MedR": null,
   "R@1": 1.35,
   "R@5": 0.46,
   "R@10": 0.23,
   "MRR": 0.84,
   "nDCG@10": 0.61
  }
 ],
 "by_group": [
  {
   "method": "clip",
   "split": "test",
   "group": "direct",
   "n": 933,
   "R@1": 56.93,
   "R@5": 84.69,
   "R@10": 93.15,
   "MRR": 69.33,
   "nDCG@10": 74.78,
   "R@1_std": 0.79,
   "R@5_std": 0.83,
   "R@10_std": 0.4,
   "MRR_std": 0.89,
   "nDCG@10_std": 0.79
  },
  {
   "method": "clip",
   "split": "test",
   "group": "indirect",
   "n": 4054,
   "R@1": 57.48,
   "R@5": 83.99,
   "R@10": 91.5,
   "MRR": 69.24,
   "nDCG@10": 74.25,
   "R@1_std": 1.02,
   "R@5_std": 0.11,
   "R@10_std": 0.38,
   "MRR_std": 0.65,
   "nDCG@10_std": 0.52
  },
  {
   "method": "clip",
   "split": "test",
   "group": "no_concept",
   "n": 13,
   "R@1": 51.35,
   "R@5": 71.88,
   "R@10": 73.84,
   "MRR": 60.87,
   "nDCG@10": 64.08,
   "R@1_std": 7.36,
   "R@5_std": 9.64,
   "R@10_std": 7.03,
   "MRR_std": 8.27,
   "nDCG@10_std": 7.98
  },
  {
   "method": "clip",
   "split": "test_full",
   "group": "direct",
   "n": 1971,
   "R@1": 48.45,
   "R@5": 76.1,
   "R@10": 84.88,
   "MRR": 60.79,
   "nDCG@10": 65.98,
   "R@1_std": 0.0,
   "R@5_std": 0.0,
   "R@10_std": 0.0,
   "MRR_std": 0.0,
   "nDCG@10_std": 0.0
  },
  {
   "method": "clip",
   "split": "test_full",
   "group": "indirect",
   "n": 8695,
   "R@1": 48.32,
   "R@5": 75.22,
   "R@10": 83.97,
   "MRR": 60.36,
   "nDCG@10": 65.46,
   "R@1_std": 0.0,
   "R@5_std": 0.0,
   "R@10_std": 0.0,
   "MRR_std": 0.0,
   "nDCG@10_std": 0.0
  },
  {
   "method": "clip",
   "split": "test_full",
   "group": "no_concept",
   "n": 30,
   "R@1": 36.67,
   "R@5": 63.33,
   "R@10": 63.33,
   "MRR": 45.47,
   "nDCG@10": 49.6,
   "R@1_std": 0.0,
   "R@5_std": 0.0,
   "R@10_std": 0.0,
   "MRR_std": 0.0,
   "nDCG@10_std": 0.0
  },
  {
   "method": "clip",
   "split": "val",
   "group": "direct",
   "n": 1003,
   "R@1": 63.7,
   "R@5": 87.69,
   "R@10": 93.39,
   "MRR": 74.05,
   "nDCG@10": 78.43,
   "R@1_std": 0.79,
   "R@5_std": 1.18,
   "R@10_std": 0.25,
   "MRR_std": 0.71,
   "nDCG@10_std": 0.51
  },
  {
   "method": "clip",
   "split": "val",
   "group": "indirect",
   "n": 3989,
   "R@1": 56.8,
   "R@5": 84.06,
   "R@10": 92.08,
   "MRR": 68.93,
   "nDCG@10": 74.18,
   "R@1_std": 0.68,
   "R@5_std": 0.6,
   "R@10_std": 0.51,
   "MRR_std": 0.56,
   "nDCG@10_std": 0.53
  },
  {
   "method": "clip",
   "split": "val",
   "group": "no_concept",
   "n": 8,
   "R@1": 47.2,
   "R@5": 54.23,
   "R@10": 61.27,
   "MRR": 51.87,
   "nDCG@10": 54.06,
   "R@1_std": 7.26,
   "R@5_std": 3.06,
   "R@10_std": 3.99,
   "MRR_std": 4.28,
   "nDCG@10_std": 3.07
  },
  {
   "method": "clip",
   "split": "val_full",
   "group": "direct",
   "n": 2100,
   "R@1": 54.0,
   "R@5": 79.52,
   "R@10": 87.33,
   "MRR": 65.19,
   "nDCG@10": 70.04,
   "R@1_std": 0.0,
   "R@5_std": 0.0,
   "R@10_std": 0.0,
   "MRR_std": 0.0,
   "nDCG@10_std": 0.0
  },
  {
   "method": "clip",
   "split": "val_full",
   "group": "indirect",
   "n": 8514,
   "R@1": 47.2,
   "R@5": 74.42,
   "R@10": 83.7,
   "MRR": 59.5,
   "nDCG@10": 64.7,
   "R@1_std": 0.0,
   "R@5_std": 0.0,
   "R@10_std": 0.0,
   "MRR_std": 0.0,
   "nDCG@10_std": 0.0
  },
  {
   "method": "clip",
   "split": "val_full",
   "group": "no_concept",
   "n": 19,
   "R@1": 47.37,
   "R@5": 52.63,
   "R@10": 52.63,
   "MRR": 50.64,
   "nDCG@10": 50.69,
   "R@1_std": 0.0,
   "R@5_std": 0.0,
   "R@10_std": 0.0,
   "MRR_std": 0.0,
   "nDCG@10_std": 0.0
  },
  {
   "method": "clip_lex",
   "split": "test",
   "group": "direct",
   "n": 933,
   "R@1": 67.37,
   "R@5": 90.65,
   "R@10": 96.36,
   "MRR": 77.7,
   "nDCG@10": 82.1,
   "R@1_std": 1.31,
   "R@5_std": 0.72,
   "R@10_std": 0.37,
   "MRR_std": 0.96,
   "nDCG@10_std": 0.66
  },
  {
   "method": "clip_lex",
   "split": "test",
   "group": "indirect",
   "n": 4054,
   "R@1": 56.44,
   "R@5": 83.66,
   "R@10": 91.57,
   "MRR": 68.52,
   "nDCG@10": 73.71,
   "R@1_std": 1.4,
   "R@5_std": 0.41,
   "R@10_std": 0.26,
   "MRR_std": 0.84,
   "nDCG@10_std": 0.63
  },
  {
   "method": "clip_lex",
   "split": "test",
   "group": "no_concept",
   "n": 13,
   "R@1": 51.35,
   "R@5": 71.88,
   "R@10": 73.84,
   "MRR": 60.87,
   "nDCG@10": 64.08,
   "R@1_std": 7.36,
   "R@5_std": 9.64,
   "R@10_std": 7.03,
   "MRR_std": 8.27,
   "nDCG@10_std": 7.98
  },
  {
   "method": "clip_lex",
   "split": "test_full",
   "group": "direct",
   "n": 1971,
   "R@1": 56.62,
   "R@5": 82.6,
   "R@10": 90.61,
   "MRR": 68.29,
   "nDCG@10": 73.31,
   "R@1_std": 0.0,
   "R@5_std": 0.0,
   "R@10_std": 0.0,
   "MRR_std": 0.0,
   "nDCG@10_std": 0.0
  },
  {
   "method": "clip_lex",
   "split": "test_full",
   "group": "indirect",
   "n": 8695,
   "R@1": 47.53,
   "R@5": 74.96,
   "R@10": 84.15,
   "MRR": 59.97,
   "nDCG@10": 65.2,
   "R@1_std": 0.0,
   "R@5_std": 0.0,
   "R@10_std": 0.0,
   "MRR_std": 0.0,
   "nDCG@10_std": 0.0
  },
  {
   "method": "clip_lex",
   "split": "test_full",
   "group": "no_concept",
   "n": 30,
   "R@1": 36.67,
   "R@5": 63.33,
   "R@10": 63.33,
   "MRR": 45.47,
   "nDCG@10": 49.6,
   "R@1_std": 0.0,
   "R@5_std": 0.0,
   "R@10_std": 0.0,
   "MRR_std": 0.0,
   "nDCG@10_std": 0.0
  }
 ],
 "alpha_curve": [
  {
   "alpha": 0.0,
   "R@1": 36.79,
   "R@5": 61.45,
   "R@10": 71.9,
   "MRR": 48.62
  },
  {
   "alpha": 0.05,
   "R@1": 36.79,
   "R@5": 61.45,
   "R@10": 71.9,
   "MRR": 48.62
  },
  {
   "alpha": 0.1,
   "R@1": 36.79,
   "R@5": 61.45,
   "R@10": 71.9,
   "MRR": 48.62
  },
  {
   "alpha": 0.15,
   "R@1": 36.81,
   "R@5": 61.48,
   "R@10": 71.93,
   "MRR": 48.63
  },
  {
   "alpha": 0.2,
   "R@1": 36.98,
   "R@5": 61.69,
   "R@10": 72.2,
   "MRR": 48.83
  },
  {
   "alpha": 0.25,
   "R@1": 37.72,
   "R@5": 62.48,
   "R@10": 72.93,
   "MRR": 49.59
  },
  {
   "alpha": 0.3,
   "R@1": 39.22,
   "R@5": 63.83,
   "R@10": 74.47,
   "MRR": 51.06
  },
  {
   "alpha": 0.35,
   "R@1": 40.95,
   "R@5": 65.63,
   "R@10": 76.22,
   "MRR": 52.74
  },
  {
   "alpha": 0.4,
   "R@1": 42.5,
   "R@5": 67.77,
   "R@10": 78.32,
   "MRR": 54.39
  },
  {
   "alpha": 0.45,
   "R@1": 44.06,
   "R@5": 69.69,
   "R@10": 80.08,
   "MRR": 55.98
  },
  {
   "alpha": 0.5,
   "R@1": 45.5,
   "R@5": 71.48,
   "R@10": 81.51,
   "MRR": 57.47
  },
  {
   "alpha": 0.55,
   "R@1": 46.85,
   "R@5": 73.22,
   "R@10": 82.71,
   "MRR": 58.84
  },
  {
   "alpha": 0.6,
   "R@1": 48.07,
   "R@5": 74.38,
   "R@10": 83.74,
   "MRR": 60.0
  },
  {
   "alpha": 0.65,
   "R@1": 49.19,
   "R@5": 75.22,
   "R@10": 84.37,
   "MRR": 60.99
  },
  {
   "alpha": 0.7,
   "R@1": 49.67,
   "R@5": 75.93,
   "R@10": 84.76,
   "MRR": 61.52
  },
  {
   "alpha": 0.75,
   "R@1": 50.17,
   "R@5": 76.34,
   "R@10": 84.92,
   "MRR": 61.98
  },
  {
   "alpha": 0.8,
   "R@1": 50.26,
   "R@5": 76.42,
   "R@10": 85.02,
   "MRR": 62.09
  },
  {
   "alpha": 0.85,
   "R@1": 50.27,
   "R@5": 76.29,
   "R@10": 85.0,
   "MRR": 62.04
  },
  {
   "alpha": 0.9,
   "R@1": 49.8,
   "R@5": 76.17,
   "R@10": 84.83,
   "MRR": 61.71
  },
  {
   "alpha": 0.95,
   "R@1": 49.21,
   "R@5": 75.72,
   "R@10": 84.65,
   "MRR": 61.2
  },
  {
   "alpha": 1.0,
   "R@1": 48.55,
   "R@5": 75.39,
   "R@10": 84.36,
   "MRR": 60.61
  }
 ],
 "rank_dist": {
  "seed0": [
   {
    "bin": "1",
    "count": 2909,
    "pct": 58.17
   },
   {
    "bin": "2",
    "count": 650,
    "pct": 13.0
   },
   {
    "bin": "3–5",
    "count": 656,
    "pct": 13.12
   },
   {
    "bin": "6–10",
    "count": 359,
    "pct": 7.18
   },
   {
    "bin": "11–20",
    "count": 275,
    "pct": 5.5
   },
   {
    "bin": "21–50",
    "count": 129,
    "pct": 2.58
   },
   {
    "bin": "51–100",
    "count": 16,
    "pct": 0.32
   },
   {
    "bin": ">100",
    "count": 7,
    "pct": 0.14
   }
  ],
  "seed1": [
   {
    "bin": "1",
    "count": 2802,
    "pct": 56.02
   },
   {
    "bin": "2",
    "count": 669,
    "pct": 13.37
   },
   {
    "bin": "3–5",
    "count": 718,
    "pct": 14.35
   },
   {
    "bin": "6–10",
    "count": 392,
    "pct": 7.84
   },
   {
    "bin": "11–20",
    "count": 287,
    "pct": 5.74
   },
   {
    "bin": "21–50",
    "count": 101,
    "pct": 2.02
   },
   {
    "bin": "51–100",
    "count": 12,
    "pct": 0.24
   },
   {
    "bin": ">100",
    "count": 21,
    "pct": 0.42
   }
  ],
  "seed2": [
   {
    "bin": "1",
    "count": 2895,
    "pct": 57.89
   },
   {
    "bin": "2",
    "count": 650,
    "pct": 13.0
   },
   {
    "bin": "3–5",
    "count": 666,
    "pct": 13.32
   },
   {
    "bin": "6–10",
    "count": 401,
    "pct": 8.02
   },
   {
    "bin": "11–20",
    "count": 249,
    "pct": 4.98
   },
   {
    "bin": "21–50",
    "count": 103,
    "pct": 2.06
   },
   {
    "bin": "51–100",
    "count": 19,
    "pct": 0.38
   },
   {
    "bin": ">100",
    "count": 18,
    "pct": 0.36
   }
  ],
  "full": [
   {
    "bin": "1",
    "count": 5167,
    "pct": 48.31
   },
   {
    "bin": "2",
    "count": 1269,
    "pct": 11.86
   },
   {
    "bin": "3–5",
    "count": 1623,
    "pct": 15.17
   },
   {
    "bin": "6–10",
    "count": 934,
    "pct": 8.73
   },
   {
    "bin": "11–20",
    "count": 823,
    "pct": 7.69
   },
   {
    "bin": "21–50",
    "count": 643,
    "pct": 6.01
   },
   {
    "bin": "51–100",
    "count": 164,
    "pct": 1.53
   },
   {
    "bin": ">100",
    "count": 73,
    "pct": 0.68
   }
  ]
 },
 "examples": [
  {
   "kind": "thất bại",
   "rank": 1784,
   "group": "no_concept",
   "query": "There is no image here to provide a caption for.",
   "query_concepts": "",
   "gt_image": "coco575410",
   "gt_objects": "stone, wall, train car, mountain, train track, house, locomotive, train, grass, train locomotive, building, tree",
   "top1_image": "coco408789",
   "top1_objects": "tv, john mccain, sign, text, tile hearth, set, television, show, man, suit",
   "gt_url": "https://cs.stanford.edu/people/rak248/VG_100K/713116.jpg",
   "top1_url": "https://cs.stanford.edu/people/rak248/VG_100K/2327812.jpg"
  },
  {
   "kind": "thất bại",
   "rank": 1694,
   "group": "indirect",
   "query": "A photo of an outside with various things in the scene.",
   "query_concepts": "outside",
   "gt_image": "coco229747",
   "gt_objects": "wing, engine, sky, plane, nose, underside, tail, light, window",
   "top1_image": "coco368994",
   "top1_objects": "this, ros, headband, flower, scarf, plate, blanket, bag, strap, mirror",
   "gt_url": "https://cs.stanford.edu/people/rak248/VG_100K_2/2393490.jpg",
   "top1_url": "https://cs.stanford.edu/people/rak248/VG_100K_2/2389949.jpg"
  },
  {
   "kind": "thất bại",
   "rank": 1361,
   "group": "indirect",
   "query": "a blue sign in front of leafy green trees",
   "query_concepts": "blue sign, green tree",
   "gt_image": "coco28156",
   "gt_objects": "hat, man, bench, tie, shirt, house",
   "top1_image": "coco186797",
   "top1_objects": "car, street sign, lettering, sign, road, tree, leaf, letter e, pole, suv",
   "gt_url": "https://cs.stanford.edu/people/rak248/VG_100K/2345225.jpg",
   "top1_url": "https://cs.stanford.edu/people/rak248/VG_100K/2364941.jpg"
  },
  {
   "kind": "thất bại",
   "rank": 1319,
   "group": "indirect",
   "query": "A full view of an outdoor space with many things to see.",
   "query_concepts": "outdoor, space",
   "gt_image": "coco273118",
   "gt_objects": "foot, wire, dirt, elephant, fence, food, tail, sky, shadow, ear, bush, trunk",
   "top1_image": "coco443887",
   "top1_objects": "graden statue, plant, pillar, cat, vase, flower pot, cement stone, brick, stone structure, wall",
   "gt_url": "https://cs.stanford.edu/people/rak248/VG_100K_2/2381672.jpg",
   "top1_url": "https://cs.stanford.edu/people/rak248/VG_100K_2/2395427.jpg"
  },
  {
   "kind": "thất bại",
   "rank": 1291,
   "group": "no_concept",
   "query": "I am unable to see the image above.",
   "query_concepts": "",
   "gt_image": "coco423576",
   "gt_objects": "dot, table, glasses, keyboard, mug, cup, stirrer, item, book, white stirrer, coffee cup, cocoa",
   "top1_image": "coco352080",
   "top1_objects": "suitcase, clothes, box, shirt, blanket, bookshelf, clothing, shelf, wall, red flag",
   "gt_url": "https://cs.stanford.edu/people/rak248/VG_100K/2373202.jpg",
   "top1_url": "https://cs.stanford.edu/people/rak248/VG_100K/2376838.jpg"
  },
  {
   "kind": "thành công",
   "rank": 1,
   "group": "indirect",
   "query": "Two cats on a desk and one wears pink feathers across the back.",
   "query_concepts": "cat, desk, feather",
   "gt_image": "coco94651",
   "gt_objects": "paper, cat, bell, desk, boa, wall, board, pink boa, black book, bulletin board, ear, bin",
   "top1_image": "coco94651",
   "top1_objects": "paper, cat, bell, desk, boa, wall, board, pink boa, black book, bulletin board",
   "gt_url": "https://cs.stanford.edu/people/rak248/VG_100K/713794.jpg",
   "top1_url": "https://cs.stanford.edu/people/rak248/VG_100K/713794.jpg"
  },
  {
   "kind": "thành công",
   "rank": 1,
   "group": "indirect",
   "query": "A sign with the word pub sits on a wall and underneath it on a table is a blender and three bottles of variety of alcohol and some flavoring.",
   "query_concepts": "sign, word, wall, table, blender, bottle",
   "gt_image": "coco214224",
   "gt_objects": "ice cube, bottle, letter, shelf, blender, wall, sign, book, cocktail shaker, lime juice, triple sec, glasses",
   "top1_image": "coco214224",
   "top1_objects": "ice cube, bottle, letter, shelf, blender, wall, sign, book, cocktail shaker, lime juice",
   "gt_url": "https://cs.stanford.edu/people/rak248/VG_100K_2/2394044.jpg",
   "top1_url": "https://cs.stanford.edu/people/rak248/VG_100K_2/2394044.jpg"
  },
  {
   "kind": "thành công",
   "rank": 1,
   "group": "indirect",
   "query": "Two bowls of creamy soup and broccoli on a wood table.",
   "query_concepts": "bowl, soup, broccoli, wood table",
   "gt_image": "coco2759",
   "gt_objects": "spoon, table, bowl, handle, napkin, hole, sauce, broccoli, soup, curve, cloth, wood",
   "top1_image": "coco2759",
   "top1_objects": "spoon, table, bowl, handle, napkin, hole, sauce, broccoli, soup, curve",
   "gt_url": "https://cs.stanford.edu/people/rak248/VG_100K/2333046.jpg",
   "top1_url": "https://cs.stanford.edu/people/rak248/VG_100K/2333046.jpg"
  },
  {
   "kind": "thành công",
   "rank": 1,
   "group": "indirect",
   "query": "a home bar with different drink ingredients under a large decorative sign saying, \"PUB\".",
   "query_concepts": "home, bar, drink, large, sign",
   "gt_image": "coco214224",
   "gt_objects": "ice cube, bottle, letter, shelf, blender, wall, sign, book, cocktail shaker, lime juice, triple sec, glasses",
   "top1_image": "coco214224",
   "top1_objects": "ice cube, bottle, letter, shelf, blender, wall, sign, book, cocktail shaker, lime juice",
   "gt_url": "https://cs.stanford.edu/people/rak248/VG_100K_2/2394044.jpg",
   "top1_url": "https://cs.stanford.edu/people/rak248/VG_100K_2/2394044.jpg"
  },
  {
   "kind": "thành công",
   "rank": 1,
   "group": "indirect",
   "query": "A sign with the word \"Pub\" hangs above several liquor bottles.",
   "query_concepts": "sign, word, bottle",
   "gt_image": "coco214224",
   "gt_objects": "ice cube, bottle, letter, shelf, blender, wall, sign, book, cocktail shaker, lime juice, triple sec, glasses",
   "top1_image": "coco214224",
   "top1_objects": "ice cube, bottle, letter, shelf, blender, wall, sign, book, cocktail shaker, lime juice",
   "gt_url": "https://cs.stanford.edu/people/rak248/VG_100K_2/2394044.jpg",
   "top1_url": "https://cs.stanford.edu/people/rak248/VG_100K_2/2394044.jpg"
  }
 ]
};
