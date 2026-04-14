.PHONY: week2-preflight

week2-preflight:
	@echo "=== Week 2 Pre-flight Checks ==="
	@echo ""
	@# 1. Check data files are writable (not root-owned)
	@echo "--- Data file ownership ---"
	@fail=0; \
	for f in data/*.csv data/*.json; do \
		if [ -e "$$f" ]; then \
			if [ ! -w "$$f" ]; then \
				echo "FAIL: $$f is not writable (possibly root-owned)"; \
				fail=1; \
			else \
				echo "  OK: $$f is writable"; \
			fi; \
		fi; \
	done; \
	if [ $$fail -eq 1 ]; then \
		echo "ACTION: fix ownership with chown or recreate files"; \
	fi
	@echo ""
	@# 2. Verify .gitignore has artifact rules
	@echo "--- .gitignore artifact rules ---"
	@if grep -q 'artifacts/fragrance_\*/embeddings.npy' .gitignore && \
	    grep -q 'artifacts/multimodal_\*/doc_embeddings.npy' .gitignore && \
	    grep -q '!artifacts/occasions/embeddings.npy' .gitignore; then \
		echo "  OK: .gitignore has artifact exclusion rules"; \
	else \
		echo "FAIL: .gitignore missing artifact exclusion rules"; \
	fi
	@echo ""
	@# 3. Check for uncommitted changes in src/vibescents/
	@echo "--- Uncommitted changes in src/vibescents/ ---"
	@if git diff HEAD -- src/vibescents/ | grep -q '^'; then \
		echo "WARN: uncommitted changes in src/vibescents/:"; \
		git diff HEAD --stat -- src/vibescents/; \
	else \
		echo "  OK: no uncommitted changes in src/vibescents/"; \
	fi
	@echo ""
	@# 4. Verify week2_pipeline import
	@echo "--- Import check: vibescents.week2_pipeline ---"
	@if uv run python -c "from vibescents import week2_pipeline" 2>/dev/null; then \
		echo "  OK: week2_pipeline imports successfully"; \
	else \
		echo "WARN: week2_pipeline not importable yet (module may not exist)"; \
	fi
	@echo ""
	@# 5. Run tests
	@echo "--- Running tests ---"
	@uv run pytest -x --tb=short
	@echo ""
	@# 6. Summary
	@echo "=== Pre-flight complete ==="
	@echo "Review any FAIL/WARN lines above before proceeding."
