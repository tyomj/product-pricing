IMAGE_NAME=product-pricing
CONTAINER_NAME=product-pricing-container

build:
	nvidia-docker build -t ${IMAGE_NAME} -f Dockerfile .

run:
	nvidia-docker run -it --rm --ipc=host --name=${CONTAINER_NAME} \
		-v ${PWD}/data:/product-pricing/data \
		-v ${PWD}/weights:/product-pricing/weights \
		-v ${PWD}/prod_pricing:/product-pricing/prod_pricing \
		${IMAGE_NAME} /bin/bash
