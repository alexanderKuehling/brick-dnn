/**
 * The pages about Brick DNN itself, in English: for lecturers (/teaching/), who is behind it
 * (/about/) and security (/security/). Every statement here is one the repository backs: the
 * plans (site.ts), the privacy policy, terms and refund policy (deploy/sites/brickdnn.com/legal/),
 * SECURITY.md and docs/. No customers, numbers of users or quotes: there are none to name yet.
 */
import {
  absolute,
  BADGE_PATHS,
  CHANGELOG_URL,
  CONTACT_EMAIL,
  DATA_PROTECTION_PATHS,
  EDITOR_PATH,
  editorExampleUrl,
  EXAMPLES_PATHS,
  OUTAGES_URL,
  PLANS,
  RELEASE,
  REPOSITORY_URL,
  SECURITY_ADVISORY_URL,
  SECURITY_PATHS,
  SECURITY_TXT_PATH,
  TEACHING_PATHS,
  UPTIME_URL,
} from '../site.ts';
import type { InfoPage } from './types.ts';

const free = PLANS.find((plan) => plan.price === 0)!;
const mail = `<a href="mailto:${CONTACT_EMAIL}">${CONTACT_EMAIL}</a>`;
const year = RELEASE.date.slice(0, 4);

/** The badge as Markdown, as docs/badge.md and the share dialog of the app write it. */
export const BADGE_MARKDOWN = `[![Open in Brick DNN](${absolute(BADGE_PATHS.en)})](${absolute(editorExampleUrl('unet'))})`;

/** How to cite the software (CITATION.cff has the same data). */
export const CITATION_TEXT = `Kühling, A. (${year}). Brick DNN (Version ${RELEASE.version}) [Computer software]. ${REPOSITORY_URL}`;
export const CITATION_BIBTEX = `@software{kuehling_brick_dnn_${year},
  author  = {Kühling, Alexander},
  title   = {Brick DNN},
  version = {${RELEASE.version}},
  year    = {${year}},
  url     = {${REPOSITORY_URL}}
}`;

const teaching: InfoPage = {
  id: 'teaching',
  title: 'Brick DNN for teaching deep learning – courses, labs and assignments',
  description:
    'For lecturers: students build networks from layers and export PyTorch code without an account. Shared networks for assignments, built-in datasets, self-hosting.',
  kicker: 'For lecturers',
  h1: 'Brick DNN in a deep-learning course',
  lead: 'Students connect layers, see the shape of every tensor and take the PyTorch code with them. Nothing to install, and no account for building and exporting.',
  action: { label: 'Open the editor', href: EDITOR_PATH },
  sections: [
    {
      id: 'course',
      title: 'What a course can rely on',
      html: `<ul class="grid two">
<li><h3>No account for students</h3><p>Building networks, checking them and exporting the code work without an account, so a lab can run without any student signing up. Training on the server needs a free account (${free.cpuHoursPerMonth} CPU hours per month).</p></li>
<li><h3>Assignments as links</h3><p>Share a network with a link as the starting point of an exercise. Whoever opens it sees the layers and the code and can open a copy in the editor. Training runs compared side by side are shared the same way, read-only, for a report or a discussion in class.</p></li>
<li><h3>Code that runs anywhere</h3><p>Every network exports as one PyTorch file, classical models as a scikit-learn script. The code imports nothing from Brick DNN and runs in a notebook, on a laptop or on the institute’s cluster.</p></li>
<li><h3>Datasets built in</h3><p>MNIST, Fashion-MNIST and CIFAR-10 for images, <em>Sentiment</em>, <em>Key lookup</em> and <em>Topics</em> for text, <em>Shapes</em> and <em>Cells</em> for segmentation. A first exercise needs no upload.</p></li>
<li><h3>Your own instance</h3><p>Brick DNN is open source under the Apache 2.0 licence and runs with Docker Compose. An institute can host it itself, switch off public sign-up and train on its own GPU.</p></li>
<li><h3>Data in Switzerland</h3><p>brickdnn.com runs on a server in Geneva, without cookies and without tracking. The <a href="${DATA_PROTECTION_PATHS.en}">fact sheet for data-protection officers</a> says what is processed in a course; the <a href="/accessibility">accessibility statement</a> says what works with a keyboard and a screen reader.</p></li>
</ul>`,
    },
    {
      id: 'badge',
      title: 'Link from your course material',
      html: `<div class="prose"><p>This badge opens one of the <a href="${EXAMPLES_PATHS.en}">built-in examples</a> in the editor. In a README or a notebook:</p>
<p><img src="${BADGE_PATHS.en}" width="166" height="28" alt="The badge: a button with the Brick DNN logo and the words “Open in Brick DNN”" loading="lazy"></p>
<pre class="snippet" tabindex="0" aria-label="Markdown of the badge"><code>${BADGE_MARKDOWN}</code></pre>
<p>For a network of your own, turn on its link under <strong>Share</strong> in the editor. The dialog has the badge for that link, as Markdown and as HTML.</p></div>`,
    },
    {
      id: 'discount',
      title: 'Plans for courses',
      html: `<div class="prose"><p>Building and exporting cost nothing, and the free plan covers small trainings. For larger ones there are <a href="/pricing">monthly plans</a>.</p></div>
<p class="discount">Discounts for teaching and research are available on request: ${mail}.</p>`,
    },
    {
      id: 'cite',
      title: 'How to cite',
      html: `<div class="prose"><p>If Brick DNN is part of course material or a publication, please cite the software:</p>
<p>${CITATION_TEXT}</p>
<pre class="snippet" tabindex="0" aria-label="The citation as BibTeX"><code>${CITATION_BIBTEX}</code></pre>
<p>The repository’s <a href="${REPOSITORY_URL}/blob/main/CITATION.cff">CITATION.cff</a> has the same data.</p></div>`,
    },
  ],
};

const about: InfoPage = {
  id: 'about',
  title: 'About Brick DNN – who builds it, where data lives, how to reach support',
  description:
    'Brick DNN is built and run by one person, an independent developer in Germany. Open source under Apache 2.0, data on a server in Switzerland, support by e-mail.',
  kicker: 'About',
  h1: 'Who is behind Brick DNN',
  lead: 'Brick DNN is built and run by one person. I am Alexander Kühling, an independent developer in Germany.',
  sections: [
    {
      id: 'facts',
      title: 'In short',
      html: `<dl class="glance">
<div><dt>One person</dt><dd>I write the software, run the server and answer the e-mails myself. The <a href="/imprint">imprint</a> (in German) has the legal details.</dd></div>
<div><dt>Open source</dt><dd>The whole application is on <a href="${REPOSITORY_URL}">GitHub</a> under the <a href="${REPOSITORY_URL}/blob/main/LICENSE">Apache 2.0 licence</a>. The <a href="${CHANGELOG_URL}">changelog</a> lists what changed and when.</dd></div>
<div><dt>Where data lives</dt><dd>On a server of Infomaniak in Geneva, Switzerland: the application, the database, uploaded files and training results. No cookies, no tracking. The <a href="/privacy">privacy policy</a> (in German) and the <a href="${SECURITY_PATHS.en}">security page</a> have the details.</dd></div>
<div><dt>Support</dt><dd>Write to ${mail}. Bugs and wishes can also go to the <a href="${REPOSITORY_URL}/issues">issue tracker</a> on GitHub.</dd></div>
<div><dt>Paid plans</dt><dd>Monthly, and cancellable at the end of each paid month. Within 14 days of your first purchase of a plan you get the full amount back on request, without giving a reason. The <a href="/refund">refund policy</a> and the <a href="/terms">terms</a> (in German) have the conditions.</dd></div>
</dl>`,
    },
    {
      id: 'continuity',
      title: 'If the service ends',
      html: `<div class="prose"><p>A fair question to ask of a service that one person runs. What you make here does not depend on brickdnn.com: exported scripts import nothing from Brick DNN, and the application is open source. If the hosted service ever has to end, I commit to this:</p>
<ul class="checks">
<li><strong>90 days’ notice.</strong> At least 90 days before the end, by e-mail to every account and on this site.</li>
<li><strong>Exports until the last day.</strong> During these 90 days everything keeps working: networks, data, results, trained weights and scripts download as before, and <em>Download my data</em> gives you all of it as a ZIP file.</li>
<li><strong>Money back for unused time.</strong> Whatever you paid for the time after the end is refunded in proportion.</li>
<li><strong>The code stays.</strong> The source code remains available under the Apache 2.0 licence, so the last version can run on your own server.</li>
</ul></div>`,
    },
  ],
};

const security: InfoPage = {
  id: 'security',
  title: 'Security at Brick DNN – logins, backups and how to report a vulnerability',
  description:
    'How brickdnn.com protects accounts and data: encrypted connections, two-factor login, hourly database backups copied to a second storage in Switzerland.',
  kicker: 'Security',
  h1: 'Security',
  lead: 'What protects accounts and data on brickdnn.com, and how to report a problem.',
  sections: [
    {
      id: 'measures',
      title: 'What is in place',
      html: `<dl class="glance">
<div><dt>Connections</dt><dd>Every connection to brickdnn.com is encrypted (HTTPS), and browsers are told to use nothing else (HSTS).</dd></div>
<div><dt>Login</dt><dd>Passwords are stored only as salted hashes. Every account can switch on two-factor login with an authenticator app and recovery codes.</dd></div>
<div><dt>Tokens</dt><dd>API tokens for scripts and AI assistants are stored as hashes and shown once. A token can be limited to reading, and the activity page lists what each token and connected app did.</dd></div>
<div><dt>Backups</dt><dd>The database is backed up every hour and every day. The backups are copied to a second, separate storage in Switzerland (Swiss Backup by Infomaniak) and deleted there after 35 days at the latest.</dd></div>
<div><dt>Training</dt><dd>A training runs code that Brick DNN generates from checked layer settings: numbers, choices and names, never Python code that a user supplies. Uploaded ZIP files are unpacked with checks against path traversal, links and oversized contents.</dd></div>
<div><dt>Server</dt><dd>One server of Infomaniak in Geneva, Switzerland, with the database on the same machine. The <a href="/privacy">privacy policy</a> (in German) names every other provider and what it receives.</dd></div>
<div><dt>These pages</dt><dd>Static HTML with a strict Content-Security-Policy. They load nothing from other servers.</dd></div>
<div><dt>Status</dt><dd>An automatic check asks the site every 15 minutes whether it answers. Its <a href="${UPTIME_URL}">runs</a> are public, and so is every <a href="${OUTAGES_URL}">outage</a> it reported.</dd></div>
</dl>`,
    },
    {
      id: 'report',
      title: 'Report a vulnerability',
      html: `<div class="prose"><p>Please do not open a public issue. Report it in private through <a href="${SECURITY_ADVISORY_URL}">GitHub security advisories</a> or by e-mail to ${mail}. You will get an answer as soon as possible.</p>
<p>The same contact is in <a href="${SECURITY_TXT_PATH}">security.txt</a>. The source code is public: <a href="${REPOSITORY_URL}">github.com/alexanderKuehling/brick-dnn</a>.</p></div>`,
    },
  ],
};

const dataProtection: InfoPage = {
  id: 'dataProtection',
  title: 'Data protection in a course with Brick DNN – fact sheet for officers',
  description:
    'For data-protection officers and IT: what brickdnn.com processes when a course uses it without student accounts, where it is stored and for how long.',
  kicker: 'Fact sheet',
  h1: 'Data protection in a course',
  lead: 'For data-protection officers and IT departments: what brickdnn.com processes when a course uses it. The <a href="/privacy">privacy policy</a> (in German) is the binding text; this page sums it up.',
  sections: [
    {
      id: 'without-accounts',
      title: 'A lab without student accounts',
      html: `<dl class="glance">
<div><dt>Accounts</dt><dd>None. Building networks, checking them and exporting the code work without signing up. The server does not learn who a student is.</dd></div>
<div><dt>Sent to the server</dt><dd>The network a student is working on, that is its layers and their settings, so that the server can check it and write its code. It is used for the answer and not stored.</dd></div>
<div><dt>Kept in the browser</dt><dd>The working copy of the network, the last settings, the language and the colour scheme, in the browser’s local storage. No cookies.</dd></div>
<div><dt>Server logs</dt><dd>The shortened IP address (the last part of an IPv4 address is 0), date and time, the page without its parameters, status, size, response time and the browser’s identifier. The logs are overwritten by size and can be a few weeks old. Counters per IP address that limit the request rate last an hour at most.</dd></div>
<div><dt>Error reports</dt><dd>Technical errors of the app go to Sentry’s EU region: the message, the page without parameters, browser and operating system, and the country derived from the IP address. No user name, no form contents; the IP address is not stored there.</dd></div>
<div><dt>Tracking</dt><dd>None. No analytics, no advertising, no content loaded from other servers.</dd></div>
</dl>`,
    },
    {
      id: 'with-accounts',
      title: 'If students train on the server',
      html: `<dl class="glance">
<div><dt>Account</dt><dd>User name, e-mail address and the password as a hash. Training on the server, saving networks and uploading data need this free account.</dd></div>
<div><dt>Content</dt><dd>The networks a student saves, uploaded files and the results of trainings. Personal data of other people does not belong in uploads; the built-in datasets need none.</dd></div>
<div><dt>E-mail</dt><dd>Only about the account: confirming the address, resetting the password, security notices. Sent through Brevo in the EU.</dd></div>
<div><dt>Deletion</dt><dd>Students delete single uploads and trainings, or the whole account, themselves. Uploads and trainings nobody used for 180 days are deleted automatically, accounts after two years without use.</dd></div>
</dl>`,
    },
    {
      id: 'where',
      title: 'Where, and who is involved',
      html: `<dl class="glance">
<div><dt>Controller</dt><dd>Alexander Kühling; the address is in the <a href="/imprint">imprint</a> (in German).</dd></div>
<div><dt>Server and database</dt><dd>One server of Infomaniak in Geneva, Switzerland, under a data processing agreement (Art. 28 GDPR). The EU recognises Switzerland’s level of data protection as adequate (Art. 45 GDPR).</dd></div>
<div><dt>Other providers</dt><dd>Sentry for error reports and Brevo for account e-mails, both under data processing agreements. Paddle only when somebody buys a plan.</dd></div>
<div><dt>Backups</dt><dd>Of the database every hour and every day, copied to a second storage in Switzerland and deleted there after 35 days at the latest.</dd></div>
<div><dt>Without brickdnn.com</dt><dd>An institute can run Brick DNN on its own server; then nothing of the above leaves the institution. The exported code runs on its machines in any case.</dd></div>
</dl>
<div class="prose"><p>Does your institution need something that is not here? Write to ${mail}. More for lecturers: <a href="${TEACHING_PATHS.en}">Brick DNN in a course</a>.</p></div>`,
    },
  ],
};

export const infoEn: InfoPage[] = [teaching, about, security, dataProtection];
