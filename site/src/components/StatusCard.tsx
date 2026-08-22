import * as React from "react";
import { useState } from "react";
import { createReactAdapter } from "@proto.ui/adapter-react";
import { shadcnButton } from "@proto.ui/prototypes-shadcn/button";

const ProtoButton = createReactAdapter({ React })(shadcnButton);

type EngineState = {
  model?: string;
  ready?: boolean;
  active_requests?: number;
  queued_requests?: number;
  pool?: { used: number; total: number; pct: number };
  vram?: { used_gb: number; total_gb: number };
  error?: string;
};

export default function StatusCard() {
  const [state, setState] = useState<EngineState | null>(null);
  const [loading, setLoading] = useState(false);

  async function refresh() {
    setLoading(true);
    try {
      const token = localStorage.getItem("fastllm_admin_token") ?? "";
      const base = localStorage.getItem("fastllm_admin_base") || "http://127.0.0.1:8000";
      const response = await fetch(`${base}/admin/api/state`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = (await response.json()) as EngineState;
      setState(typeof payload.ready === "boolean" ? payload : { error: "状态接口返回了无法识别的数据" });
    } catch (error) {
      setState({ error: error instanceof Error ? error.message : "无法连接状态接口" });
    } finally {
      setLoading(false);
    }
  }

  const hasMetrics = state?.pool && state.vram;

  return (
    <section className="status-card" aria-labelledby="engine-status-title">
      <div className="status-card-copy">
        <p className="eyebrow">Optional live check · :8000</p>
        <h2 id="engine-status-title">FastLLM 引擎状态</h2>
        <div className="status-result" aria-live="polite" aria-busy={loading}>
          {state?.error ? (
            <p className="status-error" role="alert">{state.error}</p>
          ) : state ? (
            <>
              <p><strong>{state.model || "未命名模型"}</strong> <span className={state.ready ? "status-ready" : "status-waiting"}>{state.ready ? "READY" : "LOADING"}</span></p>
              <p className="status-detail">活跃 {state.active_requests ?? "—"} · 排队 {state.queued_requests ?? "—"}</p>
              {hasMetrics && (
                <p className="status-detail">页池 {state.pool!.used}/{state.pool!.total} ({state.pool!.pct.toFixed(0)}%) · VRAM {state.vram!.used_gb.toFixed(1)}/{state.vram!.total_gb.toFixed(1)} GB</p>
              )}
            </>
          ) : (
            <p className="status-placeholder">只在浏览器中按需连接本机管理接口；静态站不会自动发起请求。</p>
          )}
        </div>
      </div>
      <ProtoButton
        variant="outline"
        size="sm"
        disabled={loading}
        onClick={refresh}
        hostClassName="pui-button pui-button-secondary"
      >
        {loading ? "正在读取…" : "读取本机状态"}
      </ProtoButton>
    </section>
  );
}
