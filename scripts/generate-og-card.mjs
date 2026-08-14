import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import sharp from 'sharp';

const root = join(import.meta.dirname, '..');
const logo = await sharp(await readFile(join(root, 'src/assets/logo.png')))
	.resize(430, 430, { fit: 'contain' })
	.png()
	.toBuffer();

const copy = Buffer.from(`
	<svg width="1200" height="630" xmlns="http://www.w3.org/2000/svg">
		<defs>
			<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
				<stop offset="0" stop-color="#101724"/>
				<stop offset="1" stop-color="#18243a"/>
			</linearGradient>
			<linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">
				<stop offset="0" stop-color="#7dd3fc"/>
				<stop offset="1" stop-color="#a78bfa"/>
			</linearGradient>
		</defs>
		<rect width="1200" height="630" fill="url(#bg)"/>
		<circle cx="1050" cy="80" r="240" fill="#38bdf8" opacity=".06"/>
		<circle cx="1110" cy="600" r="300" fill="#8b5cf6" opacity=".06"/>
		<rect x="525" y="174" width="92" height="8" rx="4" fill="url(#accent)"/>
		<text x="525" y="282" fill="#f8fafc" font-family="Arial, Helvetica, sans-serif"
			font-size="82" font-weight="700" letter-spacing="-3">pegainfer</text>
		<text x="530" y="348" fill="#cbd5e1" font-family="Arial, Helvetica, sans-serif"
			font-size="32" font-weight="400">Pure Rust + CUDA</text>
		<text x="530" y="395" fill="#cbd5e1" font-family="Arial, Helvetica, sans-serif"
			font-size="32" font-weight="400">LLM Inference Engine</text>
		<text x="530" y="478" fill="#7dd3fc" font-family="Arial, Helvetica, sans-serif"
			font-size="22" font-weight="700" letter-spacing="2">PEGAINFER.ORG</text>
	</svg>
`);

await sharp(copy)
	.composite([{ input: logo, left: 55, top: 100 }])
	.png({ compressionLevel: 9, adaptiveFiltering: true })
	.toFile(join(root, 'public/og-card.png'));
