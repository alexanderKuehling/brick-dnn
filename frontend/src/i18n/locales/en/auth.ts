/** Logging in, signing up, password reset, confirming the e-mail address. */
export const auth = {
  username: 'Username',
  password: 'Password',
  email: 'Email',
  logIn: 'Log in',
  backToLogin: 'Back to logging in',
  passwordHint: 'At least 8 characters, not too common.',
  login: {
    title: 'Welcome back',
    noAccount: 'No account yet? <anchor>Create one</anchor>',
    forgot: 'Forgot password?',
    failed: 'Login failed.',
    twoFactorTitle: 'Two-factor authentication',
    loggingInAs: 'Logging in as {{username}}',
    expired: 'The login has expired or was cancelled. Enter your password again.',
    enterRecovery: 'Enter one of your recovery codes. Each code works once.',
    enterCode: 'Enter the 6-digit code from your authenticator app.',
    recoveryCode: 'Recovery code',
    code: 'Code',
    verify: 'Verify',
    useApp: 'Use the authenticator app',
    useRecovery: 'Use a recovery code',
    codeRejected: 'The code was not accepted.',
  },
  register: {
    title: 'Create an account',
    already: 'Already registered? <anchor>Log in</anchor>',
    intro:
      'An account lets you save networks on this server. Building and exporting code works without one.',
    emailHint: 'Other people can invite you to their networks with it.',
    acceptTerms: 'I agree to the <terms>terms of use</terms>.',
    acceptTermsAndPrivacy:
      'I agree to the <terms>terms of use</terms> and have read the <privacy>privacy policy</privacy>.',
    privacyNote: 'How this server handles your data: <privacy>privacy policy</privacy>.',
    submit: 'Create account',
    failed: 'Registration failed.',
  },
  forgot: {
    title: 'Forgot your password?',
    unavailable:
      'Password reset is not available on this server: it sends no e-mail. Ask its operator for help.',
    checkMail: 'Check your e-mail',
    nothingArrived:
      'Nothing arrived? Check the spam folder. Accounts whose e-mail address was never confirmed cannot get a link; ask the operator of this site for help.',
    intro:
      'Enter your username or e-mail address. We send a link to choose a new password to the confirmed e-mail address of your account.',
    login: 'Username or e-mail address',
    submit: 'Send me a link',
  },
  reset: {
    title: 'Choose a new password',
    mismatch: 'The two passwords are not the same.',
    changedTitle: 'Password changed',
    changed:
      'Your password was changed and you were logged out everywhere. Log in with the new one.',
    invalidTitle: 'This link does not work',
    invalid: 'The link is invalid, has expired (it is valid for one hour) or was used already.',
    newLink: 'Ask for a new link',
    chooseFor: 'Choose a new password for <b>{{username}}</b>.',
    newPassword: 'New password',
    repeat: 'Repeat the new password',
    submit: 'Change password',
  },
  verify: {
    title: 'Confirm your e-mail address',
    failed: 'The address could not be confirmed.',
    thanks: 'Thank you',
    confirmedAddress:
      '{{email}} is confirmed. If you ever forget your password, you can reset it with a link sent to it.',
    confirmed:
      'Your e-mail address is confirmed. If you ever forget your password, you can reset it with a link sent to it.',
    toAccount: 'Go to your account',
    notConfirmed: 'Not confirmed',
    logOutOf: 'Log out of {{username}}',
    thisAccount: 'this account',
    sendNew: 'Send a new link from your account',
  },
  notFound: {
    title: 'Page not found',
    text: 'This page does not exist.',
    back: 'Back to the builder',
  },
  requireLogin: 'Log in to see this page.',
};

/** The legal pages (imprint, privacy policy …) the operator puts on the server. */
export const legal = {
  labels: {
    imprint: 'Imprint',
    privacy: 'Privacy policy',
    terms: 'Terms of use',
    refund: 'Refund policy',
    accessibility: 'Accessibility statement',
  },
  short: {
    imprint: 'Imprint',
    privacy: 'Privacy',
    terms: 'Terms',
    refund: 'Refunds',
    accessibility: 'Accessibility',
  },
  contact: 'Contact',
  nav: 'Legal',
  missing: 'This page has not been set up on this server.',
  lastChanged: 'Last changed {{date}}',
  allLanguages: 'Show all languages',
};
