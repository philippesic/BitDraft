import hashlib,json,re,pathlib,datetime
root=pathlib.Path.cwd()
source=pathlib.Path("/usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.h")
target=root/"results/cuda-glibc-compat/include/crt/math_functions.h"
original=source.read_text()
copied=target.read_text()
before=hashlib.sha256(target.read_bytes()).hexdigest()
assert original==copied,"private include copy differs from CUDA source before patch"
updates=[]
for name in ("rsqrt","rsqrtf"):
    pattern=rf"(\b{name}\(float x\)|\b{name}\(double x\));"
    updated,count=re.subn(pattern,r"\1 noexcept (true);",copied)
    assert count==1,(name,count)
    copied=updated
target.write_text(copied)
after=hashlib.sha256(target.read_bytes()).hexdigest()
report={"created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"source":str(source),"private_copy":str(target),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"private_before_sha256":before,"private_after_sha256":after,"replacements":["rsqrt(double x) -> rsqrt(double x) noexcept (true)","rsqrtf(float x) -> rsqrtf(float x) noexcept (true)"],"regex_counts":{"rsqrt":1,"rsqrtf":1},"system_header_modified":False}
out=root/"results/cuda-glibc-compat/header-patch-manifest.json"
out.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))

