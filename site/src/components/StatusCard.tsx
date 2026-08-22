import { useState } from "react";
import * as React from "react";
import { createReactAdapter } from "@proto.ui/adapter-react";
import { shadcnButton } from "@proto.ui/prototypes-shadcn/button";

const ProtoButton = createReactAdapter({ React })(shadcnButton);

/**
 * Proto-UI 试用组件: shadcn 风格的状态卡片。
 * 数据源: FastLLM apiserver /admin/api/state (Bearer AUTH_TOKEN)。
 * 若 Proto-UI adapter 出现问题, 请到 Proto-UI/Proto-UI 提 issue。
 */
export default function StatusCard() {
  const [state, setState] = useState<{
    model?: string; ready?: boolean; active?: number; queued?: number;
    pool?: { used: number; total: number; pct: number };
    vram?: { used_gb: number; total_gb: number };
    error?: string;
  } | null>(null);
  const [loading, setLoading] = useState(false);

  async function refresh() {
    setLoading(true);
    try {
      const token = localStorage.getItem("fastllm_admin_token") ?? "";
      const r = await fetch(`${localStorage.getItem("fastllm_admin_base") || "http://127.0.0.1:8000"}/admin/api/state`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const j = await r.json();
      setState(j.ready !== undefined ? j : { error: "unexpected payload" });
    } catch (e) {
      setState({ error: String(e) });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{
      border: "1px solid hsl(var(--border))",
      borderRadius: "calc(var(--radius) + 2px)",
      background: "hsl(var(--card))",
      padding: "16px", minWidth: 260,
    }}>
      <div style={{ color:"hsl(var(--muted-foreground))", fontSize:12, marginBottom:8 }}>
        引擎状态 · FastLLM :8002
      </div>
      <div style={{ fontSize:15, marginBottom:10 }}>
        {state?.error ? (
          <span style={{color:"hsl(0 70% 62%)"}}>{state.error}</span>
        ) : state ? (
          <>
            <strong>{state.model}</strong>{" "}
            <span style={{color: state.ready ? "hsl(var(--accent))" : "hsl(38 80% 60%)"}}>
              {state.ready ? "READY" : "LOADING"}
            </span>
            {" · "}活跃 {state.active_requests} / 排队 {state.queued_requests}
            <br />
            <small style={{color:"hsl(var(--muted-foreground))"}}>
              页池 {state.pool.used}/{state.pool.total} ({state.pool.pct.toFixed(0)}%) · VRAM{" "}
              {state.vram.used_gb.toFixed(1)}/{state.vram.total_gb.toFixed(1)} GB
            </small>
          </>
        ) : (
          <span style={{color:"hsl(var(--muted-foreground))"}}>点击右侧按钮拉取</span>
        )}
      </div>
      <ProtoButton
        variant="default"
        size="sm"
        disabled={loading}
        onClick={refresh}
        hostStyle={{
          background:"hsl(var(--primary))", color:"#0b1220", border:0,
          borderRadius:"var(--radius)", padding:"6px 14px", cursor:"pointer",
          fontWeight:600, opacity: loading?0.6:1,
        }}
      >{loading ? "加载中…" : "刷新状态"}</ProtoButton>
    </div>
  );
}
