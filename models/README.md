# Models

- `model_GRU_(H)_(D)_(d).py` - The RNN model. The nomenclature goes as:
  - (H) - Number of GRU layers stacked
  - (D) - Number of Dense Layers on top of the stacked RNN layers
  - (d) - Signifies the use of dropout in the hidden layers
 
- `prepare_input_for_kernel.py` - Combines the result from the RNN model with hand-picked features for passing as input to secondary classifiers.
- `classifiers.py` - Passes the combined result from above to SVM, Random Forest and Adaboost for prediction.
- `util.py` - Utility functions
