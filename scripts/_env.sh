# Shared by the scripts: venv executables on macOS/Linux (.venv/bin) or Windows Git Bash (.venv/Scripts).
if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python; ABD=.venv/bin/abd
elif [ -x .venv/Scripts/python.exe ]; then
  PY=.venv/Scripts/python.exe; ABD=.venv/Scripts/abd.exe
else
  echo "no .venv found; see README (Cài đặt)" >&2; exit 1
fi
