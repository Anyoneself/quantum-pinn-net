"use strict";

// Run with the bundled Node runtime and task-local MathJax on NODE_PATH.
const fs = require("node:fs");
const path = require("node:path");
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
  const text = fs.readFileSync(source, "utf8");
  const manifest = {};
  for (const block of text.matchAll(/\\\[([\s\S]*?)\\\]/g)) {
    const number = block[1].match(/\\tag\{(\d+)\}/)[1];
    let latex = block[1].replace(/\\tag\{\d+\}/, "").trim().replace(/\.$/, "");
    if (!latex.includes("\\begin{aligned}")) {
      latex = "\\begin{gathered}" + latex.replace(/,?\\qquad/g, "\\\\") + "\\end{gathered}";
    }
    const html = adaptor.outerHTML(engine.convert(latex, { display: true }));
    if (html.includes("data-mjx-error")) throw new Error(`Equation ${number}: ${html}`);
    let svg = html.slice(html.indexOf("<svg"), html.lastIndexOf("</svg>") + 6);
    const viewbox = svg.match(/viewBox="([^"]+)"/)[1].split(/\s+/).map(Number);
    const widthEm = viewbox[2] / 1000;
    const heightEm = viewbox[3] / 1000;
    const widthPx = Math.ceil(widthEm * 14 * 4);
    const heightPx = Math.ceil(heightEm * 14 * 4);
    svg = svg.replace(/width="[^"]+"/, `width="${widthPx}px"`)
      .replace(/height="[^"]+"/, `height="${heightPx}px"`)
      .replace(/currentColor/g, "#000000");
    if (!svg.includes('xmlns="http://www.w3.org/2000/svg"')) {
      svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ');
    }
    const file = path.join(output, `equation-${number}.png`);
    await sharp(Buffer.from(svg)).png().toFile(file);
    manifest[number] = {
      file, width_cm: widthEm * 10.5 / 72 * 2.54,
      height_cm: heightEm * 10.5 / 72 * 2.54,
    };
  }
  fs.writeFileSync(path.join(output, "manifest.json"), JSON.stringify(manifest, null, 2));
  console.log(`Rendered ${Object.keys(manifest).length} equations`);
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
