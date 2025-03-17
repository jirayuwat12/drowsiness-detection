sort_library:
	isort .

format_code:
	black . -l 120

format_all:
	make sort_library
	make format_code

install_production:
	pip install .

install_development:
	pip install -e ".[dev]"

test:
	pytest

run_demo:
	streamlit run ./scripts/drowsiness_detection_demo.py
