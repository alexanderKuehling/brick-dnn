/** The shape of the English texts with every text a string: the German ones must match it. */
export type Messages<T> = {
  [K in keyof T]: T[K] extends string ? string : Messages<T[K]>;
};
