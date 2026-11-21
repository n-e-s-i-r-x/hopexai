import { copyFile, mkdir, rm } from 'node:fs/promises';
import path from 'node:path';

const root = process.cwd();
const source = path.join(root, 'app', 'static');
const output = path.join(root, '.vercel-output');

await rm(output, { recursive: true, force: true });
await mkdir(path.join(output, 'static'), { recursive: true });
await copyFile(path.join(source, 'index.html'), path.join(output, 'index.html'));
await copyFile(path.join(source, 'site.css'), path.join(output, 'static', 'site.css'));
await copyFile(path.join(source, 'site.js'), path.join(output, 'static', 'site.js'));
console.log('Built HopeXAI static frontend for Vercel.');
