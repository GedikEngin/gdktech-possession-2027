# make run BUNDLE=/path/to/bundle TEAM=<team_name>   -> out/<team_name>.csv
# make test                                           -> fixture end-to-end (byte-identical twice + validator + score)
# make validate CSV=out/<team_name>.csv
BUNDLE ?= data
TEAM ?= gdkTech
OUT ?= out/$(TEAM).csv
CSV ?= $(OUT)
VENV ?= .venv

.PHONY: run test validate venv clean docker

venv: $(VENV)/bin/python
$(VENV)/bin/python:
	@GDKB_VENV=$(VENV) ./run.sh --help >/dev/null 2>&1 || true

run:
	GDKB_VENV=$(VENV) ./run.sh $(BUNDLE) $(OUT)

validate: venv
	$(VENV)/bin/python validate.py $(CSV)

test: venv
	$(VENV)/bin/python -m pytest -q tests

docker:
	docker build -t gdkb-submission .

clean:
	rm -rf out $(VENV) .pytest_cache
