# EcoLens GP-05 hackathon improvements
- [x] Secure Cloud auth + per-user scans database
- [x] Python FastAPI classifier + optional app connection
- [x] Uncertain category, error states, dashboard breakdown + recent scans
- [x] Oceanic Teal theme (replaced old #16A34A green), README, tests, browser check
- [x] Submission readiness pass (3 Nov 2026): README team details + real screenshots, Learn hub five-category cards, dashboard empty state, replace-image control, delete confirmations, E2E + Python error-path checks, 9/9 unit tests
- [x] Python ML service implemented (4 Nov 2026): MobileNetV3-Small transfer learning, TrashNet (MIT) dataset prep with leakage-free split, train/eval scripts, local inference in FastAPI (vision API now explicit fallback only), pytest suite 9/9 passed
- [ ] Run dataset download + training + evaluation (needs GPU/torch environment) — commands in backend/README.md
- [ ] Deploy the trained Python service externally and set PYTHON_CLASSIFIER_URL (needs user hosting)
