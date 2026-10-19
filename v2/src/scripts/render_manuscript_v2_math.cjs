"use strict";

const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { SVG } = require("mathjax-full/js/output/svg.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");
const { AllPackages } = require("mathjax-full/js/input/tex/AllPackages.js");
const sharp = require("sharp");

async function main() {
  const [source, output] = process.argv.slice(2);
  const adaptor = liteAdaptor();
  RegisterHTMLHandler(adaptor);
  const engine = mathjax.document("", {
    InputJax: new TeX({ packages: AllPackages }),
    OutputJax: new SVG({ fontCache: "local" }),
  });
  fs.mkdirSync(output, { recursive: true });
  const manifest = {};
  const sourceText = fs.readFileSync(source, "utf8");
  const entries = Array.from(sourceText.matchAll(/\$\$([\s\S]*?)\$\$/g), block => ({
    number: block[1].match(/\\tag\{(\d+)\}/)[1],
    latex: block[1].replace(/\\tag\{\d+\}/, "").trim(), display: true,
  }));
  for (const latex of new Set(Array.from(sourceText.matchAll(/\\\((.*?)\\\)/g), m => m[1]))) {
    entries.push({ number: "inline-" + crypto.createHash("sha256").update(latex).digest("hex").slice(0, 12),
                   latex, display: false });
  }
  for (const entry of entries) {
    const { number, display } = entry;
    let latex = entry.latex;
    if (display && !latex.includes("\\begin{aligned}")) {
      latex = "\\begin{gathered}" + latex.replace(/,?\\qquad/g, "\\\\") + "\\end{gathered}";
    }
    const html = adaptor.outerHTML(engine.convert(latex, { display }));
    if (html.includes("data-mjx-error")) throw new Error(`Invalid equation ${number}`);
    let svg = html.slice(html.indexOf("<svg"), html.lastIndexOf("</svg>") + 6);
    const box = svg.match(/viewBox="([^"]+)"/)[1].split(/\s+/).map(Number);
    const widthEm = box[2] / 1000;
    const heightEm = box[3] / 1000;
    svg = svg.replace(/width="[^"]+"/, `width="${Math.ceil(widthEm * 56)}px"`)
      .replace(/height="[^"]+"/, `height="${Math.ceil(heightEm * 56)}px"`)
      .replace(/currentColor/g, "#000000");
    if (!svg.includes('xmlns="http://www.w3.org/2000/svg"')) {
      svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ');
    }
    const file = path.join(output, `equation-${number}.png`);
    await sharp(Buffer.from(svg)).png().toFile(file);
    manifest[number] = { file, width_cm: widthEm * 10.5 / 72 * 2.54,
                        height_cm: heightEm * 10.5 / 72 * 2.54 };
  }
  fs.writeFileSync(path.join(output, "manifest.json"), JSON.stringify(manifest, null, 2));
  console.log(`Rendered ${entries.filter(e => e.display).length} display equations and inline fallbacks`);
}

main().catch(error => { console.error(error); process.exit(1); });
