# V100-perfs 文档站

Astro 静态文档站，部署到 `cyjin-yl.github.io/v100-perfs/`。

## 技术栈

- Astro 5 + React islands
- Proto-UI/Proto-UI v0.2 adapter/prototype 思路
- shadcn 风格 token 与 Proto-UI 试用组件 `StatusCard.tsx`
- GitHub Pages Actions

Proto-UI 组件用于真实文档站场景试用。遇到可复现的 protocol、adapter、
accessibility 或 host integration 问题，应在 Proto-UI/Proto-UI 提 issue，
并在本文件与 `docs/EXPERIENCE.md` 记录复现条件。

## 本地

```bash
npm install
npm run llms-txt
npm run dev
```

生产构建：`npm run build`。
