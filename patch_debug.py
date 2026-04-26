import json

def patch_notebook(nb_path):
    with open(nb_path, 'r') as f:
        nb = json.load(f)

    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "# --- Enhanced Debugging ---" in source:
                print(f"Found in {nb_path} cell {idx}")
                
                new_debug = """# --- Enhanced Debugging (IPython / Colab compatible) ---
import traceback
import IPython

def custom_exc(shell, etype, evalue, tb, tb_offset=None):
    print("\\n" + "="*60)
    print("!!! AN ERROR OCCURRED !!!")
    print("="*60)
    traceback.print_exception(etype, evalue, tb)
    print("="*60)
    print("!!! CHECK THE TRACEBACK ABOVE TO FIND THE EXACT LINE OF CODE !!!\\n")

_ipython = IPython.get_ipython()
if _ipython:
    _ipython.set_custom_exc((Exception,), custom_exc)
    _ipython.magic("xmode Verbose")  # Further enhance built-in tracebacks"""

                # Replace the old block. Find the start and end of it.
                start_idx = source.find("# --- Enhanced Debugging ---")
                
                if "sys.excepthook_set = True" in source:
                    end_idx = source.find("sys.excepthook_set = True") + len("sys.excepthook_set = True")
                else:
                    end_idx = source.find("sys.excepthook = custom_excepthook") + len("sys.excepthook = custom_excepthook")
                
                source = source[:start_idx] + new_debug + source[end_idx:]
                
                # Split back into lines
                lines = source.split('\n')
                cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
                
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)

patch_notebook('notebooks/harsh_offline_pipeline.ipynb')
patch_notebook('notebooks/harsh_week5_qwen3vl.ipynb')
