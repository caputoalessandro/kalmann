OUT_DIR = out
GEN_DIR = gen

$(OUT_DIR)/relazione.pdf: $(wildcard src/*.md) | $(OUT_DIR)
	pandoc -t latex --toc --listings --template $(CURDIR)/eisvogel.tex -o $@ $(sort $^)

$(OUT_DIR) $(GEN_DIR):
	mkdir -p $@

clean:
	rm -rf $(OUT_DIR) $(GEN_DIR)
