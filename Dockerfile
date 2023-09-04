FROM ubuntu

RUN apt-get update && apt-get install -y \
	python3 \
	python3-pip \
	python3-dev \
	build-essential \
	python3-setuptools \
	python3-venv \
	libspatialindex6 \
	optipng \
	potrace libffi-dev libxml2-dev python3-lxml python3-xcffib \
	librsvg2-bin \
	git

RUN cd / && \
	git clone https://github.com/jkenlooper/piecemaker.git && \
	cd piecemaker && \ 
	pip install --upgrade --upgrade-strategy eager -e . && \
    pip install requests

CMD /bin/bash
