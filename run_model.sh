if [ ! -d "input" ]; then
	mkdir input
	mkdir input_clean
	mkdir submission
fi

python preproc/create_glove_wiki_model.py
python preproc/train_data_tokenize.py
python preproc/test_data_tokenize.py
python preproc/train_data_augment.py
python preproc/convert_train_test_to_indices.py

python models/model_GRU_1_1.py
python models/model_GRU_3_1.py
python models/model_GRU_3_2.py
python models/model_GRU_3_2_d.py
python models/prepare_input_for_kernel.py
python models/classifiers.py