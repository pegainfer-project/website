// @ts-check
import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import starlight from '@astrojs/starlight';
import sitemap from '@astrojs/sitemap';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

const SITE = 'https://open-infer.org';

// https://astro.build/config
export default defineConfig({
	site: SITE,
	trailingSlash: 'always',
	markdown: {
		processor: unified({
			remarkPlugins: [remarkMath],
			rehypePlugins: [rehypeKatex],
		}),
	},
	integrations: [
		sitemap(),
		starlight({
			title: 'pegainfer',
			logo: { src: './src/assets/logo.png', alt: 'pegainfer' },
			favicon: '/favicon.png',
			routeMiddleware: './src/starlightRouteData.ts',
			customCss: ['./src/styles/custom.css'],
			expressiveCode: {
				styleOverrides: {
					borderRadius: '0.375rem',
					borderWidth: '1px',
					codeFontSize: '0.8125rem',
					codeLineHeight: '1.65',
					frames: {
						frameBoxShadowCssValue: 'none',
					},
				},
				plugins: [
					{
						name: 'pegainfer-plain-code',
						hooks: {
							preprocessCode: ({ codeBlock }) => {
								// uv-style: plain code blocks unless frame= is set explicitly
								if (codeBlock.metaOptions.getString('frame') === undefined) {
									codeBlock.props.frame = 'none';
								}
							},
						},
					},
				],
			},
			// Page-specific social metadata and structured data are added in
			// src/starlightRouteData.ts.
			head: [
				{
					tag: 'link',
					attrs: { rel: 'preconnect', href: 'https://fonts.googleapis.com' },
				},
				{
					tag: 'link',
					attrs: {
						rel: 'preconnect',
						href: 'https://fonts.gstatic.com',
						crossorigin: true,
					},
				},
				{
					tag: 'link',
					attrs: {
						rel: 'stylesheet',
						href: 'https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,400;0,500;0,700;1,400&family=Roboto+Mono:ital,wght@0,400;0,500;1,400&display=swap',
					},
				},
				{
					tag: 'link',
					attrs: {
						rel: 'stylesheet',
						href: 'https://cdn.jsdelivr.net/npm/katex@0.17.0/dist/katex.min.css',
					},
				},
			],
			social: [
				{
					icon: 'github',
					label: 'GitHub',
					href: 'https://github.com/pegainfer-project/pegainfer',
				},
			],
			sidebar: [
				{ label: 'Getting Started', link: '/getting-started/' },
				{
					label: 'Blogs',
					items: [
						{ label: 'All Posts', link: '/blog/' },
						{
							label: 'Weight Loading: From Safetensors to GPU',
							link: '/blog/weight-loading/',
						},
						{
							label: 'Speculative Decoding',
							link: '/blog/speculative-decoding/',
						},
						{
							label: 'See Qwen3 Decode as a CUDA Graph',
							link: '/blog/cuda-graph-export/',
						},
						{
							label: 'PegaInfer 0.1.0: Production-Grade Rust Inference',
							link: '/blog/pegainfer-010/',
						},
						{
							label: 'Co-locating Prefill and Decode',
							link: '/blog/green-ctx/',
						},
					],
				},
				{
					label: 'Models',
					items: [
						{ label: 'Qwen3 Dense', link: '/models/qwen3-4b/' },
						{ label: 'Qwen3.5 Dense', link: '/models/qwen35/' },
						{ label: 'GLM-5.2', link: '/models/glm52/' },
						{ label: 'Gemma 4 12B', link: '/models/gemma4/' },
					],
				},
			],
		}),
	],
});
