if [ ! -d "input" ]; then
	echo "'input' directory not found, create the 'input' directory with train.csv, test.csv and the pre-trained glove embeddings."
	exit 1
fi

if [ ! -d "input_clean" ]; then
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