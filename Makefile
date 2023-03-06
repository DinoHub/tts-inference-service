build:
	docker build . -t gradio-tts-inference-service:1.0.0
dev:
	docker run -p 8084:8084 --rm -it -v ${PWD}:/dory gradio-tts-inference-service:1.0.0
