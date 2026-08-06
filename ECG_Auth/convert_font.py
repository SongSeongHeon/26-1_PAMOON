import base64
from pathlib import Path

# 폰트 파일 경로
font_path = Path("static/fonts/Cafe24Simplehae-v2.0.ttf")

# 변환 결과 JS 파일 경로
output_path = Path("static/js/Cafe24Simplehae-normal.js")

# jsPDF에 등록할 이름
font_file_name = "Cafe24Simplehae-v2.0.ttf"
font_name = "Cafe24Simplehae"
font_style = "normal"

if not font_path.exists():
    raise FileNotFoundError(f"폰트 파일을 찾을 수 없습니다: {font_path}")

font_data = font_path.read_bytes()
font_base64 = base64.b64encode(font_data).decode("utf-8")

js_code = f"""
(function (jsPDFAPI) {{
  var font = "{font_base64}";

  jsPDFAPI.events.push(["addFonts", function () {{
    this.addFileToVFS("{font_file_name}", font);
    this.addFont("{font_file_name}", "{font_name}", "{font_style}");
  }}]);
}})(window.jspdf.jsPDF.API);
"""

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(js_code, encoding="utf-8")

print(f"변환 완료: {output_path}")
print(f"등록 폰트명: {font_name}")
