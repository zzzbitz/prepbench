PYTHON ?= python

.PHONY: install lint run clean-outputs

install:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m compileall agents core config evaluate llm_connect run.py

# Usage: make run CASE=1 MODE=base RUN_MODE=full MODEL=x-ai/grok-code-fast-1 FORCE=1
run:
	$(PYTHON) run.py --case $(CASE) $(if $(MODE),--mode $(MODE),) $(if $(MODEL),--model $(MODEL),) $(if $(RUN_MODE),--run_mode $(RUN_MODE),) $(if $(FORCE),--force,)

clean-outputs:
	rm -rf @output data_synthesis/output
