import * as React from "react";
import { useState } from "react";
import { createReactAdapter } from "@proto.ui/adapter-react";
import { shadcnButton } from "@proto.ui/prototypes-shadcn/button";
import {
  shadcnSwitchRoot,
  shadcnSwitchThumb,
} from "@proto.ui/prototypes-shadcn/switch";
import {
  shadcnTabsContent,
  shadcnTabsList,
  shadcnTabsRoot,
  shadcnTabsTrigger,
} from "@proto.ui/prototypes-shadcn/tabs";
import { shadcnToggle } from "@proto.ui/prototypes-shadcn/toggle";

const adapt = createReactAdapter({ React });
const Button = adapt(shadcnButton);
const SwitchRoot = adapt(shadcnSwitchRoot);
const SwitchThumb = adapt(shadcnSwitchThumb);
const TabsRoot = adapt(shadcnTabsRoot);
const TabsList = adapt(shadcnTabsList);
const TabsTrigger = adapt(shadcnTabsTrigger);
const TabsContent = adapt(shadcnTabsContent);
const Toggle = adapt(shadcnToggle);

type View = "reader" | "operator";

export default function ProtoDashboard() {
  const [dense, setDense] = useState(false);
  const [history, setHistory] = useState(true);
  const [view, setView] = useState<View>("reader");

  return (
    <section className={`proto-dashboard${dense ? " is-dense" : ""}`} aria-labelledby="dashboard-title">
      <div className="dashboard-heading">
        <div>
          <p className="eyebrow">Repository reader</p>
          <h2 id="dashboard-title">从结论进入证据</h2>
          <p className="section-copy">按阅读目的切换入口；所有历史记录都保留在仓库索引中。</p>
        </div>
        <div className="dashboard-actions" aria-label="阅读显示设置">
          <span id="history-label">包含历史资料</span>
          <SwitchRoot
            checked={history}
            onCheckedChange={({ checked }) => setHistory(checked)}
            hostClassName="pui-switch"
          >
            <SwitchThumb hostClassName="pui-switch-thumb" />
            <span className="sr-only">包含历史资料</span>
          </SwitchRoot>
          <Toggle
            active={dense}
            onActiveChange={({ active }) => setDense(active)}
            hostClassName="pui-toggle"
          >
            紧凑
          </Toggle>
        </div>
      </div>

      <TabsRoot defaultValue="overview" hostClassName="pui-tabs">
        <TabsList hostClassName="pui-tabs-list" a11yLabel="首页内容分区">
          <TabsTrigger value="overview" hostClassName="pui-tabs-trigger">实测结论</TabsTrigger>
          <TabsTrigger value="paths" hostClassName="pui-tabs-trigger">阅读路径</TabsTrigger>
          <TabsTrigger value="scope" hostClassName="pui-tabs-trigger">资料范围</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" hostClassName="pui-tabs-content">
          <div className="finding-grid">
            <article className="finding-card">
              <p className="finding-label">模型与量化</p>
              <h3>单卡 V100 32G 可生产运行</h3>
              <p>UD-Q5_K_M 与 IQ4_XS 双 profile；工具调用约束修复后三模型验证通过。</p>
            </article>
            <article className="finding-card">
              <p className="finding-label">缓存路径</p>
              <h3>三级前缀缓存命中率超过 90%</h3>
              <p>GPU 页池、内存与傲腾 NVMe 组成 L1–L3，轮转 TTFT 从全冷 22.5 秒降至亚秒。</p>
            </article>
            <article className="finding-card">
              <p className="finding-label">协议兼容</p>
              <h3>语法推进与前缀保持稳定</h3>
              <p>字符级语法推进器对齐 llama.cpp；归因头在进入引擎前剥离，避免破坏 KV 前缀。</p>
            </article>
          </div>
        </TabsContent>

        <TabsContent value="paths" hostClassName="pui-tabs-content">
          <div className="route-choice" role="group" aria-label="阅读目的">
            <Button
              variant={view === "reader" ? "default" : "outline"}
              size="sm"
              onClick={() => setView("reader")}
              hostClassName="pui-button"
            >
              按主题阅读
              {view === "reader" && <span className="sr-only">，当前选择</span>}
            </Button>
            <Button
              variant={view === "operator" ? "default" : "outline"}
              size="sm"
              onClick={() => setView("operator")}
              hostClassName="pui-button"
            >
              按运维任务阅读
              {view === "operator" && <span className="sr-only">，当前选择</span>}
            </Button>
          </div>
          {view === "reader" ? (
            <div className="path-panel" aria-live="polite">
              <h3>先浏览完整仓库，再进入文件</h3>
              <p>分类与文件系统一致，搜索结果直接打开原始路径对应的阅读页。</p>
              <a className="text-link" href="/v100-perfs/repo">打开仓库阅读器 <span aria-hidden="true">→</span></a>
            </div>
          ) : (
            <div className="path-panel" aria-live="polite">
              <h3>从部署与故障现象开始</h3>
              <p>先看快速开始，再以仓库搜索定位服务、缓存、语法约束与性能记录。</p>
              <a className="text-link" href="/v100-perfs/docs/getting-started">打开快速开始 <span aria-hidden="true">→</span></a>
            </div>
          )}
        </TabsContent>

        <TabsContent value="scope" hostClassName="pui-tabs-content">
          <div className="scope-panel">
            <p><strong>{history ? "完整索引已开启" : "当前聚焦最新资料"}</strong></p>
            <p>{history ? "Qwen3.5、Qwen3.6 与 Qwen3.8 记录均可搜索并逐文件阅读。" : "显示偏好只影响本页说明；仓库阅读器仍保留并索引全部历史资料。"}</p>
          </div>
        </TabsContent>
      </TabsRoot>
    </section>
  );
}
