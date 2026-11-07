/** Isolate an actual Chromium storage namespace while retaining native transactions. */
export function isolateChatStorage(prefix: string): () => void {
  const open = IDBFactory.prototype.open, removeDatabase = IDBFactory.prototype.deleteDatabase;
  const get = Storage.prototype.getItem, set = Storage.prototype.setItem, remove = Storage.prototype.removeItem;
  const databaseName = (name: string) => ['skulk-chat-history-v2', 'skulk-chat-documents-v1'].includes(name) ? prefix + ':' + name : name;
  const storageKey = (key: string) => ['skulk-chat-v2', 'skulk-chat'].includes(key) ? prefix + ':' + key : key;
  IDBFactory.prototype.open = function (name, version) { return open.call(this, databaseName(name), version); };
  IDBFactory.prototype.deleteDatabase = function (name) { return removeDatabase.call(this, databaseName(name)); };
  Storage.prototype.getItem = function (key) { return get.call(this, storageKey(key)); };
  Storage.prototype.setItem = function (key, value) { set.call(this, storageKey(key), value); };
  Storage.prototype.removeItem = function (key) { remove.call(this, storageKey(key)); };
  return () => {
    IDBFactory.prototype.open = open;
    IDBFactory.prototype.deleteDatabase = removeDatabase;
    Storage.prototype.getItem = get;
    Storage.prototype.setItem = set;
    Storage.prototype.removeItem = remove;
    for (const name of ['skulk-chat-history-v2', 'skulk-chat-documents-v1']) removeDatabase.call(indexedDB, databaseName(name));
    for (const key of ['skulk-chat-v2', 'skulk-chat']) remove.call(localStorage, storageKey(key));
  };
}
