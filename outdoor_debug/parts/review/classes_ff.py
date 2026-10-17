import sys; sys.argv=["x"]
sys.path.insert(0,"/tmp/rv_smoke"); sys.path.insert(0,"/home/wg19671/.claude/jobs/a0d391cb/tmp/od")
import data.vda_semantics_dataset as vsd, data
vsd.VDASemanticsDegradedVKITTIDataset = data.VKITTIDataset   # FORCED FAIL: recipe silently gets the base class
import runpy
src=open("/tmp/rv_smoke/classes.py").read().replace('Path(__file__).parent / "classes.json"','"/tmp/rv_smoke/classes_ff.json"')
exec(compile(src,"classes_copy","exec"),{"__name__":"__main__","__file__":"/tmp/rv_smoke/classes.py"})
