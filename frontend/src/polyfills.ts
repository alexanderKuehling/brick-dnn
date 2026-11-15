/**
 * Built-in functions that dependencies call and older browsers lack. Vite only rewrites newer
 * syntax for the browsers in `build.target` (vite.config.mjs, README.md "Supported browsers"),
 * it adds no missing functions. main.tsx imports this first, and vite.config.mjs puts it into
 * a chunk of its own that runs before Sentry, React and Mantine.
 *
 * Object.hasOwn: Mantine's shallowEqual calls it while rendering, so in Safari before 15.4 and
 * Chrome before 93 the app failed at once ("Object.hasOwn is not a function").
 */

/** Whether `key` is an own property of `obj` (`Object.hasOwn`, which older browsers lack). */
export function hasOwn(obj: object, key: PropertyKey): boolean {
  // eslint-disable-next-line prefer-object-has-own -- this is what Object.hasOwn falls back to
  return Object.prototype.hasOwnProperty.call(obj, key);
}

/** Adds the missing functions to `target` (the global `Object`), leaving existing ones alone. */
export function installPolyfills(target: ObjectConstructor = Object): void {
  if (typeof target.hasOwn !== 'function') {
    // Like the built-in: writable and configurable, but not enumerable.
    Object.defineProperty(target, 'hasOwn', {
      // hasOwnProperty converts its `this` like Object.hasOwn: primitives are boxed, null and
      // undefined throw a TypeError.
      value: (obj: unknown, key: PropertyKey) => hasOwn(obj as object, key),
      writable: true,
      configurable: true,
    });
  }
}

installPolyfills();
