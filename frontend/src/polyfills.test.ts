import { hasOwn, installPolyfills } from './polyfills';

describe('hasOwn', () => {
  it('is true for own properties only', () => {
    const obj = { a: 1, missing: undefined };
    expect(hasOwn(obj, 'a')).toBe(true);
    expect(hasOwn(obj, 'missing')).toBe(true);
    expect(hasOwn(obj, 'b')).toBe(false);
    // Inherited, not own.
    expect(hasOwn(obj, 'toString')).toBe(false);
    expect(hasOwn([1, 2], 0)).toBe(true);
    expect(hasOwn([1, 2], 'length')).toBe(true);
  });

  it('works for objects without a prototype or with their own hasOwnProperty', () => {
    const bare = Object.create(null) as Record<string, unknown>;
    bare.a = 1;
    expect(hasOwn(bare, 'a')).toBe(true);
    const tricky = { hasOwnProperty: () => true };
    expect(hasOwn(tricky, 'other')).toBe(false);
  });

  it('takes symbols', () => {
    const key = Symbol('key');
    expect(hasOwn({ [key]: 1 }, key)).toBe(true);
    expect(hasOwn({}, key)).toBe(false);
  });
});

describe('installPolyfills', () => {
  const native = Object.hasOwn;

  /** Removes Object.hasOwn, as in Safari before 15.4 and Chrome before 93. */
  function removeHasOwn() {
    Reflect.deleteProperty(Object, 'hasOwn');
    expect(typeof Object.hasOwn).toBe('undefined');
  }

  afterEach(() => {
    Object.defineProperty(Object, 'hasOwn', { value: native, writable: true, configurable: true });
  });

  it('installs Object.hasOwn where the browser lacks it', () => {
    removeHasOwn();

    installPolyfills();

    expect(Object.hasOwn).not.toBe(native);
    expect(Object.hasOwn({ a: 1 }, 'a')).toBe(true);
    expect(Object.hasOwn({ a: 1 }, 'toString')).toBe(false);
    // Primitives are converted like by the built-in, null and undefined throw.
    expect(Object.hasOwn('abc' as unknown as object, 'length')).toBe(true);
    expect(() => Object.hasOwn(null as unknown as object, 'a')).toThrow(TypeError);
    // Not enumerable, like the built-in.
    expect(Object.keys(Object)).not.toContain('hasOwn');
  });

  it('keeps the built-in where there is one', () => {
    installPolyfills();
    expect(Object.hasOwn).toBe(native);
  });

  it('installs itself when the module is loaded', async () => {
    removeHasOwn();
    vi.resetModules();

    await import('./polyfills');

    expect(typeof Object.hasOwn).toBe('function');
    expect(Object.hasOwn).not.toBe(native);
  });
});
