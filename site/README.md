# V100-perfs 文档站

Astro 静态文档站，部署到 `cyjin-yl.github.io/v100-perfs/`。站点把整个仓库作为阅读对象：`/repo` 提供搜索与文件系统分类，`/read/<repo-path>` 打开被索引的文本文件。

## 技术栈

- Astro 5 + React 19 islands
- `@proto.ui/adapter-react@0.2.0`
- `@proto.ui/prototypes-shadcn@0.2.0`
- GitHub Pages Actions

首页 React islands 通过 `createReactAdapter({ React })` 适配 shadcn prototypes，实际使用以下 family subpath（避免从根入口引入无关 prototype）：

- `button`：阅读路径选择与本机状态读取
- `switch`：历史资料说明开关（Root + Thumb）
- `tabs`：实测结论、阅读路径、资料范围（Root + List + Trigger + Content）
- `toggle`：紧凑阅读模式

浏览器 DOM 中的 `data-pui-root`、`data-pui-style`、`data-pui-a11y-actions` 与 `data-checked` / `data-selected` / `data-active` 状态由 Proto-UI adapter 暴露，可用于静态部署后的集成审计。视觉样式由 `src/styles/global.css` 中的站点 token 与 `.pui-*` 类提供，不依赖 Tailwind 运行时。

## 集成边界

- Proto-UI 只存在于 React island 内；Astro 页面和布局不直接调用 adapter。当前 adapter 的 prototype host 在 React SSR 输出中为空，服务器只输出 island 的语义外壳与静态说明，`client:load` hydration 后再挂载 tabs / switch / toggle；未观察到 hydration mismatch。
- `ProtoDashboard` 使用 `client:load`，因为 tabs 是首页主要导航；可选的本机状态卡使用 `client:visible`，其按钮只在卡片进入视口并 hydration 后出现，避免首屏加载第二份交互实例。
- 控件使用 prototype 自带的键盘与 ARIA 行为；站点补充可见焦点、`aria-live`、`aria-busy`、状态标签和 reduced-motion 样式。
- prototype 内含 shadcn 风格的 Tailwind token 类，但本项目没有 Tailwind 编译步骤；因此外观必须由 adapter 的 `hostClassName` 和站点 CSS 明确定义。
- 本机状态读取仅在用户按按钮后访问浏览器 `localStorage` 与管理接口；SSR 阶段不读取 `window` / `localStorage`，静态站也不会自动请求生产服务。
- 当前只使用 family subpath imports。各 adapter island 仍会带入 Proto-UI 的共享 runtime；不要为纯静态链接创建额外 island。
- Adapter 只投影 prototype 声明过的 props；任意 `aria-*` / `data-*` 不会自动透传。Switch 0.2.0 也没有命名 prop，因此本页把隐藏文本作为 Switch 子节点提供 accessible name。

## 本地运行

```bash
npm install
npm run llms-txt
npm run dev
```

生产构建：`npm run build`。

如果发现可稳定复现的 protocol、adapter、accessibility 或 host integration 问题，应先把最小复现缩减到单个 prototype + React adapter，再向 Proto-UI/Proto-UI 提交 issue；站点 CSS、错误 hydration directive 或业务状态问题不属于上游问题。
