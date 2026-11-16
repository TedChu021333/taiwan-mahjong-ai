// 牌局與 AI 都在這條背景執行緒裡跑（Pyodide＝瀏覽器裡的 Python），畫面不會卡住。
// 主畫面送 {id, cmd, args}，這裡回 {id, result} 或 {id, error}；載入進度用 {progress} 通知。
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.29.5/full/";
importScripts(PYODIDE + "pyodide.js");

let py = null;
let bridge = null;

const SETUP = `
import json
from training.numpy_model import NumpyNet
from agent.nn_agent import NetworkPolicy
from play.web import Session

_bot = NetworkPolicy(NumpyNet.from_file("model.npz"), guard=True)
_session = None

def new_session(mode, seed=None):
    global _session
    _session = Session(_bot, seed=seed, mode=mode)

def step():
    return json.dumps(_session.step(), ensure_ascii=False)

def answer(index=None, swap=None):
    _session.answer(index=index, swap=swap)
`;

async function load() {
  postMessage({ progress: "下載 Python 執行環境…" });
  py = await loadPyodide({ indexURL: PYODIDE });
  await py.loadPackage("numpy");
  postMessage({ progress: "下載規則引擎與 AI…" });
  const [sources, model] = await Promise.all([
    fetch("py.zip").then((r) => r.arrayBuffer()),
    fetch("model.npz").then((r) => r.arrayBuffer()),
  ]);
  py.unpackArchive(sources, "zip");
  py.FS.writeFile("model.npz", new Uint8Array(model));
  postMessage({ progress: "載入 AI 模型…" });
  py.runPython(SETUP);
  bridge = {
    new_session: py.globals.get("new_session"),
    step: py.globals.get("step"),
    answer: py.globals.get("answer"),
  };
}

const ready = load();

onmessage = async (event) => {
  const { id, cmd, args } = event.data;
  try {
    await ready;
    let result = null;
    if (cmd === "ready") {
      result = true;
    } else if (cmd === "new") {
      if (args.seed == null) bridge.new_session(args.mode);
      else bridge.new_session(args.mode, args.seed);
    } else if (cmd === "step") {
      result = JSON.parse(bridge.step());
    } else if (cmd === "answer") {
      if (args.swap) {
        const tiles = py.toPy(args.swap);
        bridge.answer.callKwargs({ swap: tiles });
        tiles.destroy();
      } else {
        bridge.answer(args.index);
      }
    }
    postMessage({ id, result });
  } catch (error) {
    postMessage({ id, error: String(error && error.message ? error.message : error) });
  }
};
