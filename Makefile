# Embedded Linux Gym - top-level build entry points.
#
#   make deps     install host packages (Debian/Ubuntu)
#   make base     download Buildroot and build the base board image
#   make boot     boot the base image interactively (Ctrl-A X to quit)
#   make check    boot the base image headless and verify it reaches a shell

BR_VERSION ?= 2026.02.3
BOARD      ?= aarch64-virt

ROOT     := $(CURDIR)
WORK     := $(ROOT)/work
BR_DIR   := $(WORK)/buildroot-$(BR_VERSION)
OUT      := $(WORK)/output/$(BOARD)
IMAGES   := $(OUT)/images
export BR2_DL_DIR := $(WORK)/dl

.PHONY: deps base boot check clean

deps:
	sudo apt-get install -y qemu-system-arm build-essential file wget cpio \
		unzip rsync bc perl python3 git libncurses-dev

$(BR_DIR)/Makefile:
	mkdir -p $(WORK)
	curl -sSfL https://buildroot.org/downloads/buildroot-$(BR_VERSION).tar.gz \
		| tar xz -C $(WORK)

$(OUT)/.config: boards/$(BOARD)/defconfig | $(BR_DIR)/Makefile
	$(MAKE) -C $(BR_DIR) O=$(OUT) BR2_DEFCONFIG=$(ROOT)/boards/$(BOARD)/defconfig defconfig

base: $(OUT)/.config
	$(MAKE) -C $(BR_DIR) O=$(OUT)

boot:
	python3 harness/qemu_board.py --images $(IMAGES) --interactive

check:
	python3 harness/qemu_board.py --images $(IMAGES) --run "uname -a; cat /etc/issue"

clean:
	rm -rf $(OUT)
