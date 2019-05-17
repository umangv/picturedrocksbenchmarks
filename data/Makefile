SUBDIRS = zeisel zheng green

.PHONY: subdirs clean $(SUBDIRS)

subdirs: $(SUBDIRS)
	mkdir paul

$(SUBDIRS):
	$(MAKE) -C $@

clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) clean -C $$dir; \
	done
	rm paul paul15 -r
