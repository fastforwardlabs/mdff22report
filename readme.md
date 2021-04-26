For a new report click `use this template` on github.com. Give it a title with the report number.

# Dev

Report repository template -- we use markdown now. Write markdown in the `src` directory. See `src/00a-example.md` for how to format.

To run locally, make sure you have nodejs and NPM installed and then do:

```
npm install
npm run dev
```

# Deploy

To deploy you can clone the repo make sure you have nodejs and NPM installed and do:

```
npm install
npm run deploy
```

You should also commit and push your changes (like normal) to the repo.

# General info (for debugging)

This is a node app. All of the logic to build the report is contained in the `build_html.js` file. It uses a markdown parser called `markdown-it`, if you're having markdown problems you can look at the docs for that package. It also uses several plugins for extra markdown functionality (like footnotes), you can see those plugins in the file and look up the package online to debug/figure out options.

It is definitely not the most cleanly organized file, the CSS especially is distributed a bit confusingly. But the good news is there is no outside magic (besides the markdown parser) everything used to build the report is in that file.

Mostly for new reports you'll use edit markdown files in `src`, but there are a few things hardcoded into the `build_html` file: the meta tags for social links and the google analytics code (to find them just search the file for 'meta' and 'google'). You'll need to change those in the build file in the report repo directly.

There is also the `build_pdf.js` file which will generate a PDF of the report. You need to run this file manually when you want to generate a pdf (`node build_pdf.js`). You can change the file name generated in the build file itself (it is not a big file).

## About dev

Like I said, the main part of this script is the `build_html.js` script. The `dev` command, defined in `package.json` uses several packages to automatically reload changes in the browser as you make them. Those packages may break at some point, or have compatibility issues. If that happens, you might want to look for different packages with the reload functionality -- there's nothing special about the ones being used currently.

## About deploy

The deploy script is also in `package.json`, it uses a package to deploy to the gh-pages branch. You can set a custom domain name in the settings tab on github.

