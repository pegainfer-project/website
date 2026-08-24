import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, it } from 'node:test';
import { DIST_PAGES, REDIRECTS, SITE_ROUTES } from './routes.mjs';

const distDir = join(import.meta.dirname, '..', 'dist');
const blogPosts = [
	['blog/green-ctx/index.html', '2026-06-20'],
	['blog/speculative-decoding/index.html', '2026-07-17'],
	['blog/cuda-graph-export/index.html', '2026-07-10'],
	['blog/pegainfer-010/index.html', '2026-06-13'],
	['blog/weight-loading/index.html', '2026-07-28'],
];

function getJsonLd(html) {
	return [...html.matchAll(/<script[^>]*type="application\/ld\+json"[^>]*>(.*?)<\/script>/gs)].map(
		([, json]) => JSON.parse(json),
	);
}

describe('static build output', () => {
	for (const page of DIST_PAGES) {
		it(`dist/${page} exists`, () => {
			assert.ok(existsSync(join(distDir, page)), `missing dist/${page}`);
		});
	}

	it('sitemap lists every expected route', () => {
		const sitemap = readFileSync(join(distDir, 'sitemap-0.xml'), 'utf8');
		for (const route of SITE_ROUTES) {
			const loc = `https://open-infer.org${route === '/' ? '/' : route}`;
			assert.match(sitemap, new RegExp(`<loc>${loc}</loc>`), `sitemap missing ${loc}`);
		}
	});

	it('preserves legacy URLs as permanent redirects', () => {
		const redirects = readFileSync(join(distDir, '_redirects'), 'utf8')
			.split('\n')
			.map((line) => line.trim())
			.filter((line) => line && !line.startsWith('#'));

		assert.deepEqual(
			redirects,
			REDIRECTS.map(([source, destination, status]) => `${source} ${destination} ${status}`),
		);
	});

	it('404 page is Starlight styled, not a bare error', () => {
		const html = readFileSync(join(distDir, '404.html'), 'utf8');
		assert.match(html, /Page not found/i);
		assert.match(html, /starlight|pegainfer/i);
	});

	it('Qwen3 documents both speculative drafter formats and keeps stable anchors', () => {
		const modelPage = readFileSync(join(distDir, 'models/qwen3-4b/index.html'), 'utf8');
		assert.match(modelPage, /id="dflash-and-dspark-speculative-decoding"/);
		assert.match(modelPage, /id="dspark-speculative-decoding"/);
		assert.match(
			modelPage,
			/href="https:\/\/huggingface\.co\/deepseek-ai\/dflash_qwen3_4b_block7"/,
		);
		assert.match(
			modelPage,
			/href="https:\/\/huggingface\.co\/deepseek-ai\/dspark_qwen3_4b_block7"/,
		);
		assert.match(modelPage, /supports greedy and sampled/);
		assert.doesNotMatch(modelPage, /greedy-only/);

		const blogPage = readFileSync(join(distDir, 'blog/speculative-decoding/index.html'), 'utf8');
		assert.match(
			blogPage,
			/href="https:\/\/open-infer\.org\/models\/qwen3-4b\/#dflash-and-dspark-speculative-decoding"/,
		);
	});

	it('homepage has website social metadata and SoftwareApplication structured data', () => {
		const html = readFileSync(join(distDir, 'index.html'), 'utf8');
		assert.match(html, /property="og:type" content="website"/);
		assert.match(html, /property="og:image" content="https:\/\/open-infer\.org\/og-card\.png"/);
		assert.deepEqual(
			getJsonLd(html).map((entry) => entry['@type']),
			['SoftwareApplication'],
		);
	});

	for (const [page, publishedDate] of blogPosts) {
		it(`${page} has article metadata and BlogPosting structured data`, () => {
			const html = readFileSync(join(distDir, page), 'utf8');
			assert.match(html, /property="og:type" content="article"/);
			assert.match(html, new RegExp(`property="article:published_time" content="${publishedDate}`));

			const jsonLd = getJsonLd(html);
			assert.equal(jsonLd.length, 1);
			assert.equal(jsonLd[0]['@type'], 'BlogPosting');
			assert.match(jsonLd[0].mainEntityOfPage['@id'], /^https:\/\/open-infer\.org\/blog\//);
			assert.ok(jsonLd[0].author.length > 0);
		});
	}
});
