VENV = venv
# PYTHON = python3
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

#CUDA_VISIBLE_DEVICES := 0,1,2
# Version 의미 : 어떤 build.sh 실험 결과인지

run$(FEW)shotDEVICE$(CUDA_VISIBLE_DEVICES)$(VERSION)VERSION: $(VENV)/bin/activate
	@echo "CUDA device: $(CUDA_VISIBLE_DEVICES)"
	@echo "run$(FEW)shotDEVICE$(CUDA_VISIBLE_DEVICES)"
	export CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES); \
	$(PYTHON) main.py --model_name=llama-7b --n_test_samples=500 --k=$(FEW) --log_file=$(FEW)shot$(VERSION).txt --dataset=trivia_qa --demo_seeds=0,1,2,3,4
	$(PYTHON) main.py --model_name=pythia-160m --n_test_samples=500 --k=$(FEW) --log_file=$(FEW)shot$(VERSION).txt --dataset=trivia_qa --demo_seeds=0,1,2,3,4
	$(PYTHON) main.py --model_name=pythia-410m --n_test_samples=500 --k=$(FEW) --log_file=$(FEW)shot$(VERSION).txt --dataset=trivia_qa --demo_seeds=0,1,2,3,4
	$(PYTHON) main.py --model_name=pythia-1.4b --n_test_samples=500 --k=$(FEW) --log_file=$(FEW)shot$(VERSION).txt --dataset=trivia_qa --demo_seeds=0,1,2,3,4
	$(PYTHON) main.py --model_name=pythia-2.8b --n_test_samples=500 --k=$(FEW) --log_file=$(FEW)shot$(VERSION).txt --dataset=trivia_qa --demo_seeds=0,1,2,3,4
	$(PYTHON) main.py --model_name=pythia-6.9b --n_test_samples=500 --k=$(FEW) --log_file=$(FEW)shot$(VERSION).txt --dataset=trivia_qa --demo_seeds=0,1,2,3,4

run$(FEW)shotDEVICE$(CUDA_VISIBLE_DEVICES)N_RETRIEVAL_DATA$(N_RETRIEVAL_DATA)RETRIEVAL$(ENCODER_NAME)ENCODER_NAME$(VERSION)VERSION: $(VENV)/bin/activate
	@echo "CUDA device: $(CUDA_VISIBLE_DEVICES)"
	@echo "run$(FEW)shotDEVICE$(CUDA_VISIBLE_DEVICES)N_RETRIEVAL_DATA$(N_RETRIEVAL_DATA)RETRIEVAL$(ENCODER_NAME)ENCODER_NAME"
	export CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES); \
	$(PYTHON) main.py --model_name=llama-7b --n_test_samples=500 --k=$(FEW) --log_file=$(FEW)shot$(VERSION).txt --dataset=trivia_qa --demo_seeds=0,1,2,3,4 --retrieval_based_prompt_selection --n_retrieval_data=$(N_RETRIEVAL_DATA) --ordering=low_to_high --sentence_encoder=$(ENCODER_NAME)
	$(PYTHON) main.py --model_name=pythia-160m --n_test_samples=500 --k=$(FEW) --log_file=$(FEW)shot$(VERSION).txt --dataset=trivia_qa --demo_seeds=0,1,2,3,4 --retrieval_based_prompt_selection --n_retrieval_data=$(N_RETRIEVAL_DATA) --ordering=low_to_high --sentence_encoder=$(ENCODER_NAME)
	$(PYTHON) main.py --model_name=pythia-410m --n_test_samples=500 --k=$(FEW) --log_file=$(FEW)shot$(VERSION).txt --dataset=trivia_qa --demo_seeds=0,1,2,3,4 --retrieval_based_prompt_selection --n_retrieval_data=$(N_RETRIEVAL_DATA) --ordering=low_to_high --sentence_encoder=$(ENCODER_NAME)
	$(PYTHON) main.py --model_name=pythia-1.4b --n_test_samples=500 --k=$(FEW) --log_file=$(FEW)shot$(VERSION).txt --dataset=trivia_qa --demo_seeds=0,1,2,3,4 --retrieval_based_prompt_selection --n_retrieval_data=$(N_RETRIEVAL_DATA) --ordering=low_to_high --sentence_encoder=$(ENCODER_NAME)
	$(PYTHON) main.py --model_name=pythia-2.8b --n_test_samples=500 --k=$(FEW) --log_file=$(FEW)shot$(VERSION).txt --dataset=trivia_qa --demo_seeds=0,1,2,3,4 --retrieval_based_prompt_selection --n_retrieval_data=$(N_RETRIEVAL_DATA) --ordering=low_to_high --sentence_encoder=$(ENCODER_NAME)
	$(PYTHON) main.py --model_name=pythia-6.9b --n_test_samples=500 --k=$(FEW) --log_file=$(FEW)shot$(VERSION).txt --dataset=trivia_qa --demo_seeds=0,1,2,3,4 --retrieval_based_prompt_selection --n_retrieval_data=$(N_RETRIEVAL_DATA) --ordering=low_to_high --sentence_encoder=$(ENCODER_NAME)

run: $(VENV)/bin/activate
	@echo "Practice ! : run"
	for seed in 0 1 2; do \
		export CUDA_VISIBLE_DEVICES=1; \
        $(PYTHON) main.py --model_name=pythia-410m --n_test_samples=500 --k=0 --log_file=s1.txt --dataset=trivia_qa --demo_seed=2023 --permu_seed=$$seed; \
		$(PYTHON) main.py --model_name=pythia-410m --n_test_samples=500 --k=0 --log_file=s2.txt --dataset=trivia_qa --demo_seed=2023 --retrieval_based_prompt_selection --n_retrieval_data=2000 --permu_seed=$$seed; \
    done

run0: $(VENV)/bin/activate
	@echo "Practice ! run 0"
	for seed in 0 1 2; do \
		export CUDA_VISIBLE_DEVICES=1; \
		$(PYTHON) main.py --model_name=pythia-160m --n_test_samples=500 --k=5 --log_file=s00.txt --dataset=trivia_qa --demo_seed=2023 --permu_seed=$$seed --ordering=low_to_high; \
	done

run1: $(VENV)/bin/activate
	@echo "Practice ! run 1"
	for seed in 0 1 2; do \
		export CUDA_VISIBLE_DEVICES=1; \
		$(PYTHON) main.py --model_name=pythia-410m --n_test_samples=500 --k=5 --log_file=s10.txt --dataset=trivia_qa --demo_seed=2023 --permu_seed=$$seed; \
		$(PYTHON) main.py --model_name=pythia-410m --n_test_samples=500 --k=5 --log_file=s11.txt --dataset=trivia_qa --demo_seed=2023 --permu_seed=$$seed --ordering=low_to_high; \
	done

run2: $(VENV)/bin/activate
	@echo "Practice ! run 2"
	for seed in 0 1 2; do \
		export CUDA_VISIBLE_DEVICES=1; \
		$(PYTHON) main.py --model_name=pythia-410m --n_test_samples=500 --k=5 --log_file=s20.txt --dataset=trivia_qa --demo_seed=2023 --retrieval_based_prompt_selection --n_retrieval_data=5000 --permu_seed=$$seed; \
		$(PYTHON) main.py --model_name=pythia-410m --n_test_samples=500 --k=5 --log_file=s21.txt --dataset=trivia_qa --demo_seed=2023 --retrieval_based_prompt_selection --n_retrieval_data=5000 --permu_seed=$$seed --ordering=low_to_high; \
		$(PYTHON) main.py --model_name=pythia-410m --n_test_samples=500 --k=5 --log_file=s22.txt --dataset=trivia_qa --demo_seed=2023 --retrieval_based_prompt_selection --n_retrieval_data=5000 --permu_seed=$$seed --ordering=high_to_low; \
	done

run3: $(VENV)/bin/activate
	@echo "Practice ! run 3"
	for seed in 0 1 2; do \
		export CUDA_VISIBLE_DEVICES=1; \
		$(PYTHON) main.py --model_name=pythia-410m --n_test_samples=500 --k=5 --log_file=s30.txt --dataset=trivia_qa --demo_seed=2023 --retrieval_based_prompt_selection --n_retrieval_data=10000 --permu_seed=$$seed; \
		$(PYTHON) main.py --model_name=pythia-410m --n_test_samples=500 --k=5 --log_file=s31.txt --dataset=trivia_qa --demo_seed=2023 --retrieval_based_prompt_selection --n_retrieval_data=10000 --permu_seed=$$seed --ordering=low_to_high --sentence_encoder=princeton-nlp/sup-simcse-roberta-large; \
		$(PYTHON) main.py --model_name=pythia-410m --n_test_samples=500 --k=5 --log_file=s32.txt --dataset=trivia_qa --demo_seed=2023 --retrieval_based_prompt_selection --n_retrieval_data=10000 --permu_seed=$$seed --ordering=high_to_low --sentence_encoder=princeton-nlp/sup-simcse-roberta-large; \
	done

# Example about run$(k)shot
run3shot: $(VENV)/bin/activate
	export CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES); \
	$(PYTHON) main.py --model_name=llama-7b --n_test_samples=500 --k=3 --log_file=3shot.txt --dataset=trivia_qa --demo_seed=2023
	$(PYTHON) main.py --model_name=pythia-160m --n_test_samples=500 --k=3 --log_file=3shot.txt --dataset=trivia_qa --demo_seed=2023
	$(PYTHON) main.py --model_name=pythia-410m --n_test_samples=500 --k=3 --log_file=3shot.txt --dataset=trivia_qa --demo_seed=2023
	$(PYTHON) main.py --model_name=pythia-1.4b --n_test_samples=500 --k=3 --log_file=3shot.txt --dataset=trivia_qa --demo_seed=2023
	$(PYTHON) main.py --model_name=pythia-2.8b --n_test_samples=500 --k=3 --log_file=3shot.txt --dataset=trivia_qa --demo_seed=2023
	$(PYTHON) main.py --model_name=pythia-6.9b --n_test_samples=500 --k=3 --log_file=3shot.txt --dataset=trivia_qa --demo_seed=2023


$(VENV)/bin/activate: requirements.txt
	# python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

clean:
	rm -rf __pycache__
	#rm -rf $(VENV)
	rm -rf .main.py.swp
	rm -rf .Makefile.swp
	rm -rf .build0.sh.swp
	rm -rf .build1.sh.swp
	rm -rf .build2.sh.swp