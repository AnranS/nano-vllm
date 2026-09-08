/* Offline enhancements; every chapter and source snapshot also works without JS. */
'use strict';

function kvLayout(length, position, blockSize = 256) {
  if (!Number.isInteger(length) || length < 1 || length > 2048 ||
      !Number.isInteger(position) || position < 0 || position >= length || blockSize !== 256) {
    throw new RangeError('需要 1—2048 个 token，位置须在序列范围内，块大小固定为 256。');
  }
  const physical = [7, 2, 19, 4, 11, 6, 0, 15];
  const count = Math.ceil(length / blockSize);
  const logical = Math.floor(position / blockSize);
  const offset = position % blockSize;
  return {
    length, position, count, logical, offset,
    physical: physical[logical],
    slot: physical[logical] * blockSize + offset,
    unused: count * blockSize - length,
    capacity: count * blockSize,
    blocks: physical.slice(0, count).map((id, index) => ({
      id, logical: index, used: Math.min(blockSize, length - index * blockSize),
      selected: index === logical,
    })),
  };
}

if (typeof module !== 'undefined' && module.exports) module.exports = { kvLayout };

if (typeof document !== 'undefined' && document.getElementById('kv-lab')) {
  const lengthInput = document.getElementById('kv-length');
  const positionInput = document.getElementById('kv-position');
  const result = document.getElementById('kv-result');
  function update() {
    const length = Number(lengthInput.value);
    if (!Number.isInteger(length) || length < 1 || length > 2048) {
      lengthInput.setAttribute('aria-invalid', 'true');
      result.textContent = '请输入 1—2048 之间的整数。';
      return;
    }
    lengthInput.removeAttribute('aria-invalid');
    const position = Math.min(Math.max(0, Number(positionInput.value)), length - 1);
    positionInput.max = String(length - 1);
    positionInput.value = String(position);
    document.getElementById('kv-position-value').textContent = String(position);
    const state = kvLayout(length, position);
    document.getElementById('kv-metrics').innerHTML = [
      [state.count, '占用块数'], [state.unused, '末块未使用位置'], [state.capacity, '分配的 token 容量'],
    ].map(([value, label]) => `<div><strong>${value}</strong><span>${label}</span></div>`).join('');
    document.getElementById('kv-blocks').innerHTML = state.blocks.map(block =>
      `<div class="cache-block${block.selected ? ' selected' : ''}"><strong>逻辑块 ${block.logical} → 块 ${block.id}</strong>` +
      `<div class="fill-track" aria-hidden="true"><span style="width:${block.used / 256 * 100}%"></span></div>` +
      `${block.used} / 256 个位置</div>`
    ).join('');
    result.textContent = `p = ${position} → 逻辑块 ${state.logical} → 块 ${state.physical}，偏移 ${state.offset} → slot = ${state.physical} × 256 + ${state.offset} = ${state.slot}`;
  }
  lengthInput.addEventListener('input', update);
  positionInput.addEventListener('input', update);
  update();
}
