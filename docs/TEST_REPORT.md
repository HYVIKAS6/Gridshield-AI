# GridShield AI — Automated Test Execution Report 🧪

**Document Version:** 2.0  
**Test Suite:** `test_pipeline.py`  
**Execution Environment:** Python 3.12.10 (Windows amd64)  
**Total Tests:** 14  
**Pass Rate:** 100% (14 / 14 Passed)  
**Execution Duration:** 5.8 seconds  

---

## 1. Test Summary

```text
============================= Test Session Summary =============================
test_pipeline.py::test_synthetic_data_generation_reproducibility      [PASS]
test_pipeline.py::test_attack_injection_actually_modifies_telemetry  [PASS]
test_pipeline.py::test_pipeline_leak_free_scaler_fit                  [PASS]
test_pipeline.py::test_chronological_train_test_split                 [PASS]
test_pipeline.py::test_model_training_and_evaluation_on_unseen_test  [PASS]
test_pipeline.py::test_dynamic_attack_scenario_duration               [PASS]
test_pipeline.py::test_dataframe_validation_error_handling            [PASS]
test_pipeline.py::test_telemetry_dropout_and_ramping_fdi              [PASS]
test_pipeline.py::test_incident_report_generation                     [PASS]
test_pipeline.py::test_cyber_physical_consistency_engine             [PASS]
test_pipeline.py::test_pandapower_smart_grid_simulation               [PASS]
test_pipeline.py::test_lstm_model_training_and_evaluation            [PASS]
test_pipeline.py::test_explainability_and_feature_importances         [PASS]
test_pipeline.py::test_offline_sample_dataset_integrity              [PASS]
============================== 14 passed in 5.8s ============================
```

---

## 2. Detailed Verification Outcomes

1. **`test_synthetic_data_generation_reproducibility`**: Confirmed identical DataFrames when seed is constant and divergence when seed changes.
2. **`test_attack_injection_actually_modifies_telemetry`**: Confirmed attack modification modifies numerical sensor values and sets `target = 1`.
3. **`test_pipeline_leak_free_scaler_fit`**: Verified mean and variance parameters are isolated strictly to the training fold.
4. **`test_chronological_train_test_split`**: Verified timestamps in test split strictly succeed all training timestamps without chronological overlap.
5. **`test_model_training_and_evaluation_on_unseen_test`**: Verified all models (Isolation Forest, Random Forest, XGBoost, MLP) output valid metrics and prediction arrays.
6. **`test_cyber_physical_consistency_engine`**: Verified physical violation flags trigger when active load deviates from $V \cdot I \cdot \cos\theta / 1000$ and RoCoF exceeds $0.50\text{ Hz/s}$.
7. **`test_pandapower_smart_grid_simulation`**: Verified AC Newton-Raphson solver converges on IEEE 14-bus test case and produces bus voltage and line loading frames.
8. **`test_lstm_model_training_and_evaluation`**: Verified PyTorch LSTM sequence model trains, evaluates, and outputs bounded anomaly probabilities.
9. **`test_explainability_and_feature_importances`**: Verified normalized feature importance sums to $1.0$ and SHAP/contribution rankings identify primary deviation features.
10. **`test_offline_sample_dataset_integrity`**: Verified offline sample dataset in `data/sample/` is present, valid, and contains 1,200 records.
