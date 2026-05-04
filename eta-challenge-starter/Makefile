VPS ?= anton
REMOTE_DIR ?= /home/palash/gobblecube-arena/eta-challenge-starter

export VPS
export REMOTE_DIR

deploy:
	git push
	ssh $(VPS) 'cd /home/palash/gobblecube-arena && git pull'

train-full:
	./remote-train.sh

tail:
	@log=$$(awk '{print $$3}' .remote-last-run 2>/dev/null) && \
	vps=$$(awk '{print $$1}' .remote-last-run 2>/dev/null) && \
	ssh "$$vps" "tail -f $$log"

score:
	@vps=$$(awk '{print $$1}' .remote-last-run 2>/dev/null) && \
	dir=$$(awk '{print $$2}' .remote-last-run 2>/dev/null) && \
	ssh "$$vps" "cd $$dir && python grade.py"

pull:
	@vps=$$(awk '{print $$1}' .remote-last-run 2>/dev/null) && \
	dir=$$(awk '{print $$2}' .remote-last-run 2>/dev/null) && \
	rsync -av "$$vps:$$dir/model.pkl" . && \
	rsync -av "$$vps:$$dir/zone_pair_means.pkl" .

.PHONY: deploy train-full tail score pull
