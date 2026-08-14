/** Public URLs that must not 404 (trailing slash matches Starlight output). */
export const SITE_ROUTES = [
	'/',
	'/getting-started/',
	'/blog/',
	'/blog/weight-loading/',
	'/blog/speculative-decoding/',
	'/blog/cuda-graph-export/',
	'/blog/pegainfer-010/',
	'/blog/green-ctx/',
	'/models/qwen3-4b/',
	'/models/qwen35/',
	'/models/glm52/',
];

export const DIST_PAGES = [
	'index.html',
	'404.html',
	'getting-started/index.html',
	'blog/index.html',
	'blog/weight-loading/index.html',
	'blog/speculative-decoding/index.html',
	'blog/cuda-graph-export/index.html',
	'blog/pegainfer-010/index.html',
	'blog/green-ctx/index.html',
	'models/qwen3-4b/index.html',
	'models/qwen35/index.html',
	'models/glm52/index.html',
];

export const REDIRECTS = [
	['/models/qwen35-4b', '/models/qwen35/', 301],
	['/models/qwen35-4b/', '/models/qwen35/', 301],
	['/blog/openinfer-010', '/blog/pegainfer-010/', 301],
	['/blog/openinfer-010/', '/blog/pegainfer-010/', 301],
];
