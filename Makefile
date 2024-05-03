export PROJECT_NAME := $$(basename $$(pwd))
export PROJECT_VERSION := $(shell cat VERSION)
export RELEASE_VERSION := $(shell cat RELEASE_VERSION)

commit:
		git commit -am "Version $(shell cat VERSION)"
		git push
patch:
		bumpversion --allow-dirty patch
minor:
		bumpversion --allow-dirty minor
major:
		bumpversion --allow-dirty major
setup:
		python setup.py sdist
push:
		$(eval REV_FILE := $(shell ls -tr dist/*.gz | tail -1))
		twine upload $(REV_FILE)
pypi: setup push
test:
		python -m pytest tests/test_1.py
