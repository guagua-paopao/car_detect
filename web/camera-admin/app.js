(() => {
  'use strict';

  const API = '/api/v1';
  const REFRESH_MS = 5000;
  const SNAPSHOT_REFRESH_MS = 250;
  const activeRunStates = new Set(['queued', 'starting', 'running', 'reconnecting', 'stopping']);
  const state = {
    token: '',
    cameras: [],
    profiles: [],
    events: [],
    modelStatus: null,
    activeView: 'realtime',
    selectedCamera: '',
    refreshTimer: 0,
    snapshotTimer: 0,
    snapshotLoading: false,
    imageUrls: {snapshot: '', event: ''}
  };

  const $ = (id) => document.getElementById(id);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[ch]);

  function notify(message, error = false) {
    const node = $('notice');
    node.textContent = message;
    node.className = `notice${error ? ' error' : ''}`;
    clearTimeout(notify.timer);
    notify.timer = setTimeout(() => node.classList.add('hidden'), 6500);
  }

  function toast(message) {
    const node = $('toast');
    node.textContent = message;
    node.classList.add('show');
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => node.classList.remove('show'), 2200);
  }

  function headers(json = false, version = null) {
    const result = {Authorization: `Bearer ${state.token}`};
    if (json) result['Content-Type'] = 'application/json';
    if (version !== null) result['If-Match'] = `"${version}"`;
    return result;
  }

  async function api(path, options = {}) {
    const request = {...options, headers: {...headers(Boolean(options.body)), ...(options.headers || {})}};
    const response = await fetch(`${API}${path}`, request);
    const contentType = response.headers.get('content-type') || '';
    const body = response.status === 204 ? null : contentType.includes('json')
      ? await response.json() : await response.text();
    if (!response.ok) {
      const message = body && typeof body === 'object'
        ? body.error || body.error_code || `${response.status} ${response.statusText}`
        : `${response.status} ${response.statusText}`;
      throw new Error(message);
    }
    return body;
  }

  async function publicJson(path) {
    const response = await fetch(`${API}${path}`);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error || body.error_code || `${response.status} ${response.statusText}`);
    return body;
  }

  function number(value, digits = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed.toFixed(digits) : '0';
  }

  function time(value) {
    const timestamp = Number(value || 0);
    return timestamp > 0 ? new Date(timestamp).toLocaleString('zh-CN', {hour12: false}) : '—';
  }

  function clock(value) {
    const timestamp = Number(value || 0);
    return timestamp > 0 ? new Date(timestamp).toLocaleTimeString('zh-CN', {hour12: false}) : '—';
  }

  function badge(status, label = status) {
    const value = String(status || 'unknown');
    const kind = ['running', 'ready', 'ok', 'stopped', 'delivered'].includes(value)
      ? 'ok' : ['failed', 'error', 'dead_letter'].includes(value) ? 'bad' : 'warn';
    return `<span class="badge ${kind}">${escapeHtml(label || value)}</span>`;
  }

  function currentCamera() {
    return state.cameras.find((camera) => camera.camera_id === state.selectedCamera) || null;
  }

  function isCameraActive(camera) {
    return activeRunStates.has(camera?.current_run?.status || '');
  }

  function normalizeAttribute(value, fallback, confidence, stable, samples) {
    if (value && typeof value === 'object') {
      return {
        label: value.label || fallback,
        confidence: Number(value.confidence || 0),
        stable: Boolean(value.stable),
        samples: Number(value.samples_used || 0)
      };
    }
    return {label: value || fallback, confidence: Number(confidence || 0), stable: Boolean(stable), samples: Number(samples || 0)};
  }

  function normalizeTrack(track) {
    const vehicle = track.vehicle_class && typeof track.vehicle_class === 'object'
      ? track.vehicle_class : {label: track.vehicle_class, confidence: track.vehicle_class_confidence};
    const attributes = track.attributes || {};
    return {
      cameraId: track.camera_id || state.selectedCamera,
      runId: track.run_id || '',
      trackId: track.track_id ?? '—',
      state: track.state || 'unknown',
      lastSeen: track.last_seen_at_ms || 0,
      vehicleClass: vehicle.label || 'unknown',
      confidence: Number(vehicle.confidence || 0),
      body: normalizeAttribute(attributes.body_type ?? track.body_type, 'unknown', attributes.body_type_confidence ?? track.body_type_confidence, attributes.body_type_stable ?? track.body_type_stable, attributes.body_type_samples_used ?? track.body_type_samples_used),
      color: normalizeAttribute(attributes.color ?? track.color, 'unknown', attributes.color_confidence ?? track.color_confidence, attributes.color_stable ?? track.color_stable, attributes.color_samples_used ?? track.color_samples_used)
    };
  }

  function normalizeEvent(event) {
    const vehicle = event.vehicle_class && typeof event.vehicle_class === 'object'
      ? event.vehicle_class : {label: event.vehicle_class, confidence: event.vehicle_class_confidence};
    const attrs = event.attributes || {};
    return {
      ...event,
      vehicleClass: vehicle.label || 'unknown',
      confidence: Number(vehicle.confidence || 0),
      body: normalizeAttribute(attrs.body_type ?? event.body_type, 'unknown', attrs.body_type_confidence ?? event.body_type_confidence, attrs.body_type_stable ?? event.body_type_stable, attrs.body_type_samples_used ?? event.body_type_samples_used),
      color: normalizeAttribute(attrs.color ?? event.color, 'unknown', attrs.color_confidence ?? event.color_confidence, attrs.color_stable ?? event.color_stable, attrs.color_samples_used ?? event.color_samples_used)
    };
  }

  function colorClass(label) {
    const value = String(label || '').toLowerCase();
    if (value.includes('white')) return 'white';
    if (value.includes('blue')) return 'blue';
    if (value.includes('gray') || value.includes('silver')) return 'gray';
    if (value.includes('black')) return 'black';
    return 'other';
  }

  function widthClass(value) {
    const rounded = Math.max(0, Math.min(100, Math.round(Number(value || 0) / 5) * 5));
    return `bar-w-${rounded}`;
  }

  function vehicleIcon() {
    return '<svg viewBox="0 0 32 20" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><path d="M3 13 6 6h20l3 7v4H3z"/><path d="M7 6 9 3h14l2 3M3 13h26M8 17h.1M24 17h.1" stroke-linecap="round"/></svg>';
  }

  function syncSelectors() {
    const selectors = [$('monitor-camera')];
    const options = state.cameras.map((camera) => `<option value="${escapeHtml(camera.camera_id)}">${escapeHtml(camera.name || camera.camera_id)} · ${escapeHtml(camera.camera_id)}</option>`).join('');
    selectors.forEach((selector) => {
      if (!selector) return;
      const previous = selector.value || state.selectedCamera;
      selector.innerHTML = options || '<option value="">暂无摄像头</option>';
      selector.value = state.cameras.some((camera) => camera.camera_id === previous)
        ? previous : (state.cameras[0]?.camera_id || '');
    });
    if (!state.selectedCamera || !state.cameras.some((camera) => camera.camera_id === state.selectedCamera)) {
      state.selectedCamera = $('monitor-camera')?.value || state.cameras[0]?.camera_id || '';
    }
    if ($('monitor-camera')) $('monitor-camera').value = state.selectedCamera;
    const profileSelect = $('camera-profile');
    if (profileSelect) profileSelect.innerHTML = state.profiles.filter((profile) => profile.enabled !== false).map((profile) => `<option value="${escapeHtml(profile.profile_id)}">${escapeHtml(profile.display_name || profile.profile_id)} · ${escapeHtml(profile.profile_id)}</option>`).join('');
  }

  async function loadProfiles() {
    const result = await api('/camera-profiles');
    state.profiles = result.items || [];
    syncSelectors();
  }

  async function loadCameras() {
    const result = await api('/cameras');
    state.cameras = result.items || [];
    syncSelectors();
    if (state.activeView === 'cameras') renderCameraPage();
    renderCameraKpi();
  }

  function renderCameraKpi() {
    const online = state.cameras.filter(isCameraActive).length;
    if ($('kpi-cameras')) $('kpi-cameras').innerHTML = `${online}<small>/ ${state.cameras.length}</small>`;
  }

  async function loadModelStatus() {
    state.modelStatus = await api('/models/status');
    renderModelStatus();
    if (state.activeView === 'models') renderModelPage();
  }

  async function loadEvents() {
    if (!state.cameras.length) {
      state.events = [];
      renderEvents();
      return;
    }
    const responses = await Promise.all(state.cameras.map(async (camera) => {
      try {
        const result = await api(`/cameras/${encodeURIComponent(camera.camera_id)}/vehicle-events?limit=30&offset=0`);
        return result.items || [];
      } catch (error) {
        return [];
      }
    }));
    state.events = responses.flat().sort((left, right) => Number(right.occurred_at_ms || 0) - Number(left.occurred_at_ms || 0));
    renderEvents();
    if (state.activeView === 'events') renderEventsPage();
  }

  async function loadAuthenticatedImage(path) {
    const response = await fetch(`${API}${path}`, {headers: headers()});
    if (!response.ok) return false;
    const previousUrl = state.imageUrls.snapshot;
    state.imageUrls.snapshot = URL.createObjectURL(await response.blob());
    const image = $('analysis-snapshot');
    image.src = state.imageUrls.snapshot;
    image.classList.remove('hidden');
    $('snapshot-placeholder').classList.add('hidden');
    if (previousUrl) URL.revokeObjectURL(previousUrl);
    return true;
  }

  function resetSnapshot() {
    const image = $('analysis-snapshot');
    image.removeAttribute('src');
    image.classList.add('hidden');
    $('snapshot-placeholder').classList.remove('hidden');
    if (state.imageUrls.snapshot) URL.revokeObjectURL(state.imageUrls.snapshot);
    state.imageUrls.snapshot = '';
  }

  async function refreshSnapshotFrame(cameraId = state.selectedCamera) {
    if (!state.token || !cameraId || state.activeView !== 'realtime' || state.snapshotLoading) return;
    state.snapshotLoading = true;
    try {
      const encoded = encodeURIComponent(cameraId);
      const analysis = await loadAuthenticatedImage(
        `/cameras/${encodeURIComponent(cameraId)}/analysis-snapshot`);
      if (!analysis) await loadAuthenticatedImage(`/cameras/${encoded}/latest-frame`);
    } finally {
      state.snapshotLoading = false;
    }
  }

  function scheduleSnapshotStream() {
    clearInterval(state.snapshotTimer);
    state.snapshotTimer = 0;
    if (!state.token || !state.selectedCamera || state.activeView !== 'realtime') return;
    refreshSnapshotFrame().catch(() => {});
    state.snapshotTimer = setInterval(() => refreshSnapshotFrame().catch(() => {}), SNAPSHOT_REFRESH_MS);
  }

  function renderCategoryBars(tracks) {
    const counts = new Map();
    tracks.forEach((track) => counts.set(track.vehicleClass, (counts.get(track.vehicleClass) || 0) + 1));
    const preferred = ['car', 'suv', 'truck', 'van'];
    const labels = [...preferred, ...[...counts.keys()].filter((label) => !preferred.includes(String(label).toLowerCase()))].slice(0, 5);
    const max = Math.max(1, ...labels.map((label) => counts.get(label) || 0));
    $('category-bars').innerHTML = labels.length ? labels.map((label) => `<div class="category-row"><span>${escapeHtml(label)}</span><div class="bar"><i class="${widthClass(((counts.get(label) || 0) / max) * 100)}"></i></div><span>${counts.get(label) || 0}</span></div>`).join('') : '<div class="empty-state compact">暂无车辆分类数据</div>';
  }

  function renderVehicles(tracks) {
    const query = ($('monitor-search')?.value || '').trim().toLowerCase();
    const filtered = tracks.filter((track) => !query || `${track.trackId} ${track.vehicleClass} ${track.body.label} ${track.color.label}`.toLowerCase().includes(query));
    $('vehicle-count').textContent = `${filtered.length} 条轨迹`;
    $('vehicle-empty').classList.toggle('hidden', filtered.length > 0);
    $('vehicle-rows').innerHTML = filtered.map((track) => `<tr><td class="track-id">${escapeHtml(track.trackId)}</td><td><span class="vehicle-label">${vehicleIcon()}${escapeHtml(track.vehicleClass)}</span></td><td>${escapeHtml(track.body.label)}</td><td><i class="color-dot ${colorClass(track.color.label)}"></i>${escapeHtml(track.color.label)}</td><td>${number(track.confidence * 100, 1)}%</td><td>${badge(track.state === 'confirmed' || track.body.stable || track.color.stable ? 'ok' : 'waiting', track.state || 'unknown')}</td><td>${escapeHtml(clock(track.lastSeen))}</td></tr>`).join('');
  }

  function renderEvents() {
    const items = state.events.slice(0, 5).map(normalizeEvent);
    $('event-list').innerHTML = items.length ? items.map((event) => `<div class="event"><div class="event-thumb"><span>${vehicleIcon()}</span></div><div><div class="event-title">车辆经过 · ${escapeHtml(event.color.label)} ${escapeHtml(event.body.label)}</div><div class="event-meta">${escapeHtml(event.camera_id || '—')} · track ${escapeHtml(event.track_id ?? '—')}</div></div><span class="event-time">${escapeHtml(clock(event.occurred_at_ms))}</span></div>`).join('') : '<div class="empty-state compact">暂无车辆事件</div>';
    if ($('kpi-events')) $('kpi-events').innerHTML = `${state.events.length}<small>件</small>`;
  }

  function renderModelStatus() {
    const status = state.modelStatus;
    if (!status) return;
    $('model-registry-version').textContent = status.registry_version ? `Registry ${status.registry_version}` : (status.enabled === false ? '车辆分析未启用' : '模型状态');
    if (!status.items?.length) {
      $('model-grid').innerHTML = '<div class="empty-state compact">模型注册表暂无可用模型</div>';
      return;
    }
    $('model-grid').innerHTML = status.items.slice(0, 4).map((item) => `<div class="model-item"><div class="model-name">${escapeHtml(item.role || item.artifact_id)}</div><div class="model-value">${escapeHtml(item.artifact_id)}</div><div class="model-state ${item.ready ? '' : 'warn'}"><i></i>${item.ready ? '就绪' : escapeHtml(item.delivery_status || '待校验')}</div></div>`).join('');
  }

  async function handleEventAction(target) {
    const eventId = target.dataset.eventId;
    if (!eventId) return;
    $('event-detail-grid').innerHTML = '<div class="empty-state compact">正在读取事件详情…</div>';
    $('event-snapshot').classList.add('hidden');
    $('event-snapshot-placeholder').textContent = '事件快照加载中…';
    $('event-dialog').showModal();
    const detailResult = await api(`/vehicle-events/${encodeURIComponent(eventId)}`);
    const event = normalizeEvent(detailResult.event || detailResult);
    $('event-detail-grid').innerHTML = `<div><small>Event ID</small><strong>${escapeHtml(event.event_id || eventId)}</strong></div><div><small>发生时间</small><strong>${escapeHtml(time(event.occurred_at_ms))}</strong></div><div><small>摄像头 / Track</small><strong>${escapeHtml(event.camera_id || '—')} / ${escapeHtml(event.track_id ?? '—')}</strong></div><div><small>车辆类别</small><strong>${escapeHtml(event.vehicleClass)}</strong></div><div><small>车身类型</small><strong>${escapeHtml(event.body.label)} · ${number(event.body.confidence * 100, 1)}% · ${event.body.stable ? '稳定' : '观察中'}（${event.body.samples} 样本）</strong></div><div><small>颜色</small><strong>${escapeHtml(event.color.label)} · ${number(event.color.confidence * 100, 1)}% · ${event.color.stable ? '稳定' : '观察中'}（${event.color.samples} 样本）</strong></div><div><small>投递状态</small><strong>${badge(event.delivery?.status || 'unknown')}</strong></div>`;
    if (state.imageUrls.event) URL.revokeObjectURL(state.imageUrls.event);
    state.imageUrls.event = '';
    try {
      const response = await fetch(`${API}/vehicle-events/${encodeURIComponent(eventId)}/snapshot`, {headers: headers()});
      if (!response.ok) throw new Error('snapshot-not-ready');
      state.imageUrls.event = URL.createObjectURL(await response.blob());
      $('event-snapshot').src = state.imageUrls.event;
      $('event-snapshot').classList.remove('hidden');
      $('event-snapshot-placeholder').classList.add('hidden');
    } catch (error) {
      $('event-snapshot-placeholder').textContent = '事件快照暂不可用，详情仍来自后端事件记录。';
    }
  }

  function renderStatus(status, response) {
    const tracks = response.available ? (response.items || []).map(normalizeTrack) : [];
    const average = tracks.length ? tracks.reduce((sum, track) => sum + track.confidence, 0) / tracks.length : 0;
    $('kpi-vehicles').innerHTML = `${tracks.length}<small>辆</small>`;
    $('kpi-confidence').innerHTML = `${tracks.length ? number(average * 100, 1) : '—'}<small>%</small>`;
    $('overview-total').textContent = tracks.length || '0';
    $('overview-stability').textContent = tracks.length ? `● ${number(average * 100, 1)}%` : '—';
    $('overview-window').textContent = response.available ? `更新于 ${clock(response.generated_at_ms)}` : '实时 Reader 不可用';
    renderCategoryBars(tracks);
    renderVehicles(tracks);
    const live = response.available && status && ['running', 'reconnecting', 'starting'].includes(status.status);
    $('live-status').className = `live-status${live ? ' live' : ''}`;
    $('live-status').innerHTML = `<i></i>${live ? '直播中' : response.available ? '暂无运行' : '不可用'}`;
    $('snapshot-camera').textContent = currentCamera()?.name || state.selectedCamera || '—';
    $('snapshot-time').textContent = response.generated_at_ms ? time(response.generated_at_ms) : '—';
    if (!response.available) notify('车辆实时 Reader 暂未提供数据，页面未伪造实时轨迹。', true);
  }

  async function loadRealtime() {
    if (!state.token || state.activeView !== 'realtime') return;
    const cameraId = state.selectedCamera;
    if (!cameraId) {
      renderStatus(null, {available: false, items: [], generated_at_ms: 0});
      return;
    }
    const encoded = encodeURIComponent(cameraId);
    const [response, status] = await Promise.all([
      api(`/cameras/${encoded}/vehicles/realtime`),
      api(`/cameras/${encoded}/status`).catch(() => null)
    ]);
    renderStatus(status, response);
    await refreshSnapshotFrame(cameraId);
    await loadEvents();
    $('last-updated').textContent = `最后更新：${new Date().toLocaleTimeString('zh-CN', {hour12: false})}`;
  }

  function scheduleRealtime() {
    clearInterval(state.refreshTimer);
    state.refreshTimer = 0;
    if (state.activeView === 'realtime' && state.token && $('monitor-auto').checked) {
      state.refreshTimer = setInterval(() => loadRealtime().catch((error) => notify(error.message, true)), REFRESH_MS);
    }
  }

  const descriptions = {
    realtime: ['实时检测', '连接后查看摄像头车辆轨迹、属性识别与运行状态。'],
    events: ['车辆事件', '按时间、摄像头和车辆属性查询已落库的车辆经过事件。'],
    cameras: ['摄像头管理', '管理摄像头 Profile、运行状态、抽帧策略与车辆分析开关。'],
    devices: ['设备状态', '查看采集、推理、存储和数据库等运行组件的健康状态。'],
    models: ['模型管理', '查看车辆检测、属性识别模型的版本、后端和交付状态。'],
    rules: ['规则配置', '配置检测阈值、轨迹确认和属性稳定判定规则。'],
    settings: ['系统设置', '调整工作台显示、数据保留和服务运行参数。'],
    logs: ['日志管理', '按级别和来源查看服务运行日志与模型事件记录。']
  };

  function pageShell(view, actions, content) {
    const [title, description] = descriptions[view];
    return `<div class="subview"><div class="subview-head"><div><h3>${title}</h3><p>${description}</p></div><div class="subview-actions">${actions}</div></div>${content}</div>`;
  }

  function renderEventsPage() {
    const rows = state.events.map(normalizeEvent);
    const cameraOptions = state.cameras.map((camera) => `<option value="${escapeHtml(camera.camera_id)}">${escapeHtml(camera.name || camera.camera_id)}</option>`).join('');
    $('page-view').innerHTML = pageShell('events', '<button class="button" data-page-action="export-events">导出当前结果</button><button class="button primary" data-page-action="refresh-page">刷新</button>', `<div class="subview-kpis"><div class="card subview-kpi"><small>事件总数</small><strong>${state.events.length}</strong></div><div class="card subview-kpi"><small>当前摄像头</small><strong>${state.selectedCamera ? 1 : 0}</strong></div><div class="card subview-kpi"><small>待投递回调</small><strong>${rows.filter((event) => event.delivery?.status === 'pending').length}</strong></div><div class="card subview-kpi"><small>已加载摄像头</small><strong>${state.cameras.length}</strong></div></div><div class="card subview-card"><div class="subview-actions event-filter"><label class="field"><span>摄像头</span><select id="event-camera-filter"><option value="all">全部摄像头</option>${cameraOptions}</select></label><label class="field"><span>车辆类别</span><select id="event-class-filter"><option value="all">全部类别</option><option value="car">car</option><option value="suv">suv</option><option value="truck">truck</option><option value="van">van</option></select></label><label class="field"><span>Track ID</span><input id="event-track-filter" placeholder="例如 42"></label><span class="toolbar-spacer"></span><button class="button primary" data-page-action="filter-events">查询</button></div><div class="subview-table"><table><thead><tr><th>事件时间</th><th>Event ID</th><th>摄像头</th><th>Track ID</th><th>车辆类别</th><th>车身 / 颜色</th><th>置信度</th><th>投递</th><th>操作</th></tr></thead><tbody id="event-page-rows">${rows.map((event) => `<tr data-camera="${escapeHtml(event.camera_id)}" data-class="${escapeHtml(event.vehicleClass)}" data-track="${escapeHtml(event.track_id)}"><td>${escapeHtml(time(event.occurred_at_ms))}</td><td><code>${escapeHtml(event.event_id)}</code></td><td>${escapeHtml(event.camera_id)}</td><td class="track-id">${escapeHtml(event.track_id)}</td><td>${escapeHtml(event.vehicleClass)}</td><td>${escapeHtml(event.body.label)} / ${escapeHtml(event.color.label)}</td><td>${number(event.confidence * 100, 1)}%</td><td>${badge(event.delivery?.status || 'unknown')}</td><td><button class="button" data-event-action="details" data-event-id="${escapeHtml(event.event_id)}">详情</button></td></tr>`).join('')}</tbody></table><div class="empty-state ${rows.length ? 'hidden' : ''}" id="event-page-empty">当前没有符合条件的车辆事件</div></div></div>`);
  }

  function renderCameraPage() {
    const rows = state.cameras.map((camera) => {
      const active = isCameraActive(camera);
      const status = camera.current_run?.status || (camera.enabled ? 'idle' : 'stopped');
      return `<tr><td><strong>${escapeHtml(camera.name || camera.camera_id)}</strong><br><small class="muted">${escapeHtml(camera.camera_id)}</small></td><td><code>${escapeHtml(camera.camera_profile)}</code></td><td>RTSP / deployment profile</td><td>${escapeHtml(String(camera.analysis?.target_infer_fps || camera.target_infer_fps || '—'))} FPS</td><td>${camera.analysis?.enabled || camera.analysis_enabled ? badge('ok', '已启用') : badge('waiting', '已停用')}</td><td>${badge(status, status === 'running' ? '运行中' : status)}</td><td><div class="row-actions"><button class="button" data-camera-action="edit" data-camera-id="${escapeHtml(camera.camera_id)}">修改</button><button class="button" data-camera-action="status" data-camera-id="${escapeHtml(camera.camera_id)}">状态</button><button class="button ${active ? '' : 'primary'}" data-camera-action="toggle" data-camera-id="${escapeHtml(camera.camera_id)}" data-active="${active}">${active ? '停止' : '启动'}</button><button class="button danger-button" data-camera-action="delete" data-camera-id="${escapeHtml(camera.camera_id)}">删除</button></div></td></tr>`;
    }).join('');
    $('page-view').innerHTML = pageShell('cameras', '<button class="button" data-page-action="refresh-page">刷新</button><button class="button primary" data-page-action="new-camera">新增摄像头</button>', `<div class="subview-kpis"><div class="card subview-kpi"><small>摄像头总数</small><strong>${state.cameras.length}</strong></div><div class="card subview-kpi"><small>在线运行</small><strong class="success-text">${state.cameras.filter(isCameraActive).length}</strong></div><div class="card subview-kpi"><small>车辆分析启用</small><strong>${state.cameras.filter((camera) => camera.analysis?.enabled || camera.analysis_enabled).length}</strong></div><div class="card subview-kpi"><small>异常 / 停止</small><strong class="warning-text">${state.cameras.filter((camera) => !isCameraActive(camera)).length}</strong></div></div><div class="card subview-card"><h4>摄像头实例</h4><div class="subview-table"><table><thead><tr><th>Camera ID / 名称</th><th>Profile</th><th>来源</th><th>目标 FPS</th><th>车辆分析</th><th>状态</th><th>操作</th></tr></thead><tbody>${rows || '<tr><td colspan="7" class="empty-state">暂无摄像头实例</td></tr>'}</tbody></table></div></div>`);
  }

  async function renderDevicePage() {
    const [health, ready, metrics, hubs] = await Promise.all([
      publicJson('/health').catch((error) => ({success: false, error: error.message})),
      publicJson('/ready').catch((error) => ({ready: false, error: error.message})),
      api('/operations/metrics').catch(() => null),
      api('/camera-hubs').catch(() => null)
    ]);
    const healthy = Boolean(health.success);
    const readyState = Boolean(ready.ready);
    const hubItems = hubs?.items || [];
    $('page-view').innerHTML = pageShell('devices', `<span class="badge ${healthy && readyState ? 'ok' : 'warn'}">● ${healthy && readyState ? '系统正常' : '需要关注'}</span><button class="button" data-page-action="refresh-page">刷新状态</button>`, `<div class="card subview-card"><h4>服务健康</h4><div class="health-grid"><div class="health-item"><header><strong>HTTP Health</strong>${badge(healthy ? 'ok' : 'error', healthy ? '正常' : '异常')}</header><div class="progress-line ${healthy ? 'teal meter-high' : 'amber meter-low'}"><i></i></div><p>${escapeHtml(health.error || 'Redis、Camera Task 和存储检查')}</p></div><div class="health-item"><header><strong>Worker Ready</strong>${badge(readyState ? 'ok' : 'error', readyState ? '就绪' : '未就绪')}</header><div class="progress-line ${readyState ? 'teal meter-high' : 'amber meter-low'}"><i></i></div><p>${escapeHtml(ready.error || `存活 Worker：${ready.alive_workers ?? '—'} / ${ready.required_workers ?? '—'}`)}</p></div><div class="health-item"><header><strong>车辆推理运行时</strong>${badge(ready.inference_pool_ready ? 'ok' : 'waiting', ready.inference_pool_ready ? '正常' : '等待')}</header><div class="progress-line ${ready.inference_pool_ready ? 'meter-medium' : 'meter-low'}"><i></i></div><p>由统一 Camera Pipeline 提供分析状态</p></div><div class="health-item"><header><strong>运维指标</strong>${badge(metrics ? 'ok' : 'waiting', metrics ? '可读取' : '未读取')}</header><div class="progress-line teal ${metrics ? 'meter-medium' : 'meter-low'}"><i></i></div><p>车辆事件、回调和观察记录统计</p></div><div class="health-item"><header><strong>共享采集 Hub</strong>${badge(hubs ? 'ok' : 'waiting', hubs ? '可读取' : '未读取')}</header><div class="progress-line teal ${hubs ? 'meter-medium' : 'meter-low'}"><i></i></div><p>已读取 Hub：${hubItems.length} 个，状态来自 camera-hubs 接口</p></div></div></div><div class="subview-grid"><div class="card subview-card"><h4>运行摘要</h4><div class="subview-list"><div class="subview-list-row"><div><strong>活动 Camera Run</strong><small>当前启用的摄像头实例</small></div><strong>${state.cameras.filter(isCameraActive).length} / ${state.cameras.length}</strong></div><div class="subview-list-row"><div><strong>最近快照</strong><small>实时检测画面由鉴权接口返回</small></div><strong>${$('last-updated').textContent.replace('最后更新：', '')}</strong></div></div></div><div class="card subview-card"><h4>可解释状态</h4><p class="status-copy">${readyState ? 'Worker 和 Camera Pipeline 已满足就绪门槛。' : '当前环境尚未满足全部就绪门槛，页面保留后端返回的真实状态。'}</p></div></div>`);
  }

  function renderModelPage() {
    const status = state.modelStatus || {};
    const rows = (status.items || []).map((item) => `<tr><td><code>${escapeHtml(item.artifact_id)}</code></td><td>${escapeHtml(item.role)}</td><td>${escapeHtml(item.backend)}</td><td>${escapeHtml(item.precision)}</td><td>${escapeHtml(item.delivery_status)}</td><td>${item.engine_sha256 ? `<code>${escapeHtml(String(item.engine_sha256).slice(0, 12))}...</code>` : '—'}</td><td>${badge(item.ready ? 'ok' : 'waiting', item.ready ? '就绪' : '待校验')}</td></tr>`).join('');
    $('page-view').innerHTML = pageShell('models', '<button class="button" data-page-action="refresh-page">刷新注册表</button>', `<div class="subview-grid"><div class="card subview-card"><h4>已登记模型</h4><div class="subview-table"><table><thead><tr><th>Artifact</th><th>角色</th><th>后端</th><th>精度</th><th>交付状态</th><th>Engine SHA256</th><th>状态</th></tr></thead><tbody>${rows || '<tr><td colspan="7" class="empty-state">模型注册表暂无内容</td></tr>'}</tbody></table></div></div><div class="card subview-card"><h4>模型注册表摘要</h4><div class="subview-list"><div class="subview-list-row"><div><strong>Registry 版本</strong><small>只读模型注册表</small></div><strong>${escapeHtml(status.registry_version || '—')}</strong></div><div class="subview-list-row"><div><strong>标签版本</strong><small>车辆类别和属性标签</small></div><strong>${escapeHtml(status.labels_version || '—')}</strong></div><div class="subview-list-row"><div><strong>全部就绪</strong><small>需要检测和属性模型均满足门槛</small></div>${badge(status.all_ready ? 'ok' : 'waiting', status.all_ready ? '是' : '否')}</div></div></div></div>`);
  }

  function renderStaticPage(view) {
    if (view === 'rules') {
      $('page-view').innerHTML = pageShell(view, '<button class="button" data-page-action="reset-rules">恢复默认</button><button class="button primary" data-page-action="save-rules">保存规则</button>', '<div class="subview-grid"><div class="card subview-card"><h4>车辆分析规则</h4><div class="switch-row"><div><strong>启用车辆检测</strong><small>对 Camera Pipeline 的最新帧运行一级车辆检测</small></div><label class="switch"><input type="checkbox" checked><span></span></label></div><div class="switch-row"><div><strong>启用属性识别</strong><small>对高质量车辆裁剪运行车身类型和颜色识别</small></div><label class="switch"><input type="checkbox" checked><span></span></label></div><div class="switch-row"><div><strong>轨迹退出时发布事件</strong><small>每条已确认轨迹最多发布一个 vehicle_passage</small></div><label class="switch"><input type="checkbox" checked><span></span></label></div><div class="switch-row"><div><strong>低质量样本降级</strong><small>属性不稳定时保留 unknown 结果</small></div><label class="switch"><input type="checkbox" checked><span></span></label></div></div><div class="card subview-card"><h4>阈值与队列</h4><div class="form-layout"><label>检测置信度<input value="0.40"></label><label>目标推理 FPS<input value="8"></label><label>最少确认帧<input value="3"></label><label>轨迹超时（毫秒）<input value="1500"></label><label>车身类型稳定阈值<input value="0.75"></label><label>颜色稳定阈值<input value="0.70"></label><label class="full">属性队列最大长度<input value="128"></label></div></div></div>');
      return;
    }
    if (view === 'settings') {
      $('page-view').innerHTML = pageShell(view, '<button class="button" data-page-action="reset-settings">取消修改</button><button class="button primary" data-page-action="save-settings">保存设置</button>', '<div class="subview-grid"><div class="card subview-card"><h4>管理端设置</h4><div class="form-layout"><label>工作台名称<input value="车辆智能检测中心"></label><label>默认时间范围<select><option>近 1 小时</option><option>近 6 小时</option><option>今日</option></select></label><label>默认摄像头<select><option>全部摄像头</option></select></label><label>列表每页条数<select><option>20</option><option>50</option><option>100</option></select></label><label class="full">管理员 Token<input type="password" placeholder="由服务端环境变量提供，不在页面保存"></label></div></div><div class="card subview-card"><h4>数据与存储</h4><div class="form-layout"><label>快照保留天数<input value="7"></label><label>属性观测保留天数<input value="7"></label><label>JPEG 质量<input value="90"></label><label>输出模式<select><option>latest</option><option>archive</option><option>both</option></select></label><label class="full">运行输出目录<input value="由服务端配置提供" readonly></label></div></div></div>');
      return;
    }
    $('page-view').innerHTML = pageShell(view, '<button class="button" data-page-action="refresh-page">刷新日志</button>', '<div class="card subview-card"><div class="subview-actions event-filter"><label class="field"><span>级别</span><select><option>全部级别</option><option>INFO</option><option>WARN</option><option>ERROR</option></select></label><label class="field"><span>来源</span><select><option>全部来源</option><option>camera_pipeline</option><option>vehicle_runtime</option><option>callback_worker</option></select></label><label class="field"><span>关键字</span><input placeholder="搜索 Track ID / Run ID / 错误码"></label></div><div class="log-list"><div class="log-row"><time>11:10:23.184</time><span class="badge ok">INFO</span><span class="log-source">vehicle_runtime</span><span class="log-message">CAM-001 confirmed_tracks=12 infer_fps=8.0</span></div><div class="log-row"><time>11:10:18.902</time><span class="badge ok">INFO</span><span class="log-source">vehicle_event</span><span class="log-message">published vehicle_passage event=vehicle_event track=T-1042</span></div><div class="log-row"><time>11:09:59.410</time><span class="badge warn">WARN</span><span class="log-source">camera_pipeline</span><span class="log-message">source reconnect attempt=1 reason=stale_frame_timeout</span></div><div class="log-row"><time>11:06:12.663</time><span class="badge bad">ERROR</span><span class="log-source">callback_worker</span><span class="log-message">delivery retry=2 http_status=502</span></div></div></div>');
  }

  async function renderPage(view) {
    if (!state.token && !['rules', 'settings', 'logs'].includes(view)) {
      $('page-view').innerHTML = '<div class="card empty-state">请先输入管理员 Token 并连接管理端。</div>';
      return;
    }
    if (view === 'events') {
      await loadEvents();
      renderEventsPage();
    } else if (view === 'cameras') {
      await loadCameras();
      renderCameraPage();
    } else if (view === 'devices') {
      await renderDevicePage();
    } else if (view === 'models') {
      if (!state.modelStatus) await loadModelStatus();
      renderModelPage();
    } else {
      renderStaticPage(view);
    }
  }

  async function switchView(view) {
    state.activeView = view;
    clearInterval(state.refreshTimer);
    state.refreshTimer = 0;
    const realtime = view === 'realtime';
    $('realtime-view').hidden = !realtime;
    $('page-view').hidden = realtime;
    $('page-title').textContent = descriptions[view][0];
    $('page-description').textContent = descriptions[view][1];
    $$('.nav-item').forEach((item) => item.classList.toggle('active', item.dataset.view === view));
    $$('.top-nav button').forEach((item) => item.classList.toggle('active', item.dataset.topView === view));
    if (realtime) {
      await loadRealtime();
      scheduleRealtime();
      scheduleSnapshotStream();
    } else {
      clearInterval(state.snapshotTimer);
      state.snapshotTimer = 0;
      await renderPage(view);
    }
  }

  function openCameraDialog(camera = null) {
    $('camera-dialog-title').textContent = camera ? '修改摄像头' : '新增摄像头';
    $('camera-id').value = camera?.camera_id || '';
    $('camera-id').disabled = Boolean(camera);
    $('camera-version').value = camera?.version || '';
    $('camera-name').value = camera?.name || '';
    $('camera-profile').value = camera?.camera_profile || state.profiles[0]?.profile_id || '';
    $('camera-interval').value = camera?.frame_interval_ms ?? 1000;
    $('camera-output').value = camera?.output_mode || 'latest';
    $('camera-retention').value = camera?.retention_days ?? 7;
    $('camera-enabled').checked = camera?.enabled ?? true;
    $('analysis-enabled').checked = camera?.analysis?.enabled ?? camera?.analysis_enabled ?? true;
    $('analysis-fps').value = camera?.analysis?.target_infer_fps ?? camera?.target_infer_fps ?? 8;
    $('algorithm-profile').value = camera?.analysis?.algorithm_profile || camera?.algorithm_profile || 'vehicle_default';
    const selected = new Set(camera?.analysis?.algorithms || camera?.algorithms || ['vehicle_detection', 'vehicle_attribute']);
    $$('input[name="algorithm"]', $('camera-form')).forEach((input) => { input.checked = selected.has(input.value); });
    $('camera-dialog').showModal();
  }

  async function saveCamera(event) {
    event.preventDefault();
    const id = $('camera-id').value.trim();
    const editing = $('camera-id').disabled;
    const algorithms = $$('input[name="algorithm"]:checked', $('camera-form')).map((input) => input.value);
    if ($('analysis-enabled').checked && !algorithms.length) throw new Error('启用车辆分析时至少选择一个算法。');
    const payload = {
      name: $('camera-name').value.trim(), camera_profile: $('camera-profile').value,
      enabled: $('camera-enabled').checked, frame_interval_ms: Number($('camera-interval').value),
      output_mode: $('camera-output').value, jpeg_quality: 90, max_width: 0, max_height: 0,
      retention_days: Number($('camera-retention').value), max_saved_frames: 100000,
      analysis: {enabled: $('analysis-enabled').checked, target_infer_fps: Number($('analysis-fps').value), algorithm_profile: $('algorithm-profile').value.trim(), algorithms},
      callback_profile: ''
    };
    if (!editing) payload.camera_id = id;
    const options = {method: editing ? 'PATCH' : 'POST', headers: headers(true, editing ? Number($('camera-version').value) : null), body: JSON.stringify(payload)};
    if (!editing) options.headers['Idempotency-Key'] = `camera-${id}-${Date.now()}`;
    await api(editing ? `/cameras/${encodeURIComponent(id)}` : '/cameras', options);
    $('camera-dialog').close();
    await loadCameras();
    if (state.activeView === 'cameras') renderCameraPage();
    notify(editing ? '摄像头配置已更新。' : '摄像头已创建。');
  }

  async function handleCameraAction(target) {
    const id = target.dataset.cameraId;
    const camera = state.cameras.find((item) => item.camera_id === id);
    if (!camera) return;
    if (target.dataset.cameraAction === 'edit') {
      const result = await api(`/cameras/${encodeURIComponent(id)}`);
      openCameraDialog(result.camera || result);
    } else if (target.dataset.cameraAction === 'status') {
      const status = await api(`/cameras/${encodeURIComponent(id)}/status`);
      notify(`${id} 当前状态：${status.status || 'unknown'}，车辆分析：${status.analysis?.state || 'unknown'}`);
    } else if (target.dataset.cameraAction === 'toggle') {
      const active = target.dataset.active === 'true';
      await api(`/cameras/${encodeURIComponent(id)}/${active ? 'stop' : 'start'}`, {method: 'POST'});
      await loadCameras();
      renderCameraPage();
      notify(`${id} ${active ? '停止' : '启动'}命令已提交。`);
    } else if (target.dataset.cameraAction === 'delete') {
      if (!window.confirm(`确认删除摄像头 ${id}？`)) return;
      const latestResult = await api(`/cameras/${encodeURIComponent(id)}`);
      const latest = latestResult.camera || latestResult;
      await api(`/cameras/${encodeURIComponent(id)}`, {method: 'DELETE', headers: headers(false, Number(latest.version || 0))});
      await loadCameras();
      renderCameraPage();
      notify(`${id} 已删除。`);
    }
  }

  async function connect() {
    const token = $('token').value.trim();
    if (!token) { notify('请输入管理员 Token。', true); return; }
    state.token = token;
    try {
      await Promise.all([loadProfiles(), loadCameras(), loadModelStatus()]);
      await loadEvents();
      $('connection').className = 'badge ok';
      $('connection').textContent = '已连接';
      notify('车辆检测工作台已连接，Token 仅保存在当前页面内存。');
      await switchView(state.activeView);
    } catch (error) {
      state.token = '';
      $('connection').className = 'badge bad';
      $('connection').textContent = '连接失败';
      notify(error.message, true);
    }
  }

  async function connectFromFragment() {
    const fragment = new URLSearchParams(window.location.hash.replace(/^#/, ''));
    const token = fragment.get('token')?.trim();
    if (!token) return;
    window.history.replaceState(null, '', `${window.location.pathname}${window.location.search}`);
    $('token').value = token;
    await connect();
    $('token').value = '';
  }

  document.addEventListener('click', async (event) => {
    const target = event.target.closest('button');
    if (!target) return;
    try {
      if (target.dataset.view) await switchView(target.dataset.view);
      else if (target.dataset.topView) await switchView(target.dataset.topView);
      else if (target.id === 'connect') await connect();
      else if (target.id === 'notice-button') toast('当前没有新的车辆模型或摄像头告警。');
      else if (target.id === 'collapse-sidebar') document.querySelector('.app-shell').classList.toggle('sidebar-collapsed');
      else if (target.dataset.viewAction) await switchView(target.dataset.viewAction);
      else if (target.dataset.eventAction === 'details') await handleEventAction(target);
      else if (target.dataset.cameraAction) await handleCameraAction(target);
      else if (target.dataset.pageAction === 'new-camera') openCameraDialog();
      else if (target.dataset.pageAction === 'refresh-page') await renderPage(state.activeView);
      else if (target.dataset.pageAction === 'refresh-realtime') await loadRealtime();
      else if (target.dataset.pageAction === 'filter-events') filterEventPage();
      else if (target.dataset.pageAction === 'export-events') toast('事件导出需要接入导出任务接口。');
      else if (target.dataset.pageAction === 'save-rules' || target.dataset.pageAction === 'save-settings') toast('配置已保存（当前为前端原型，尚未写入服务端）。');
      else if (target.dataset.pageAction === 'reset-rules' || target.dataset.pageAction === 'reset-settings') toast('已恢复当前页面初始值。');
      else if (target.hasAttribute('data-close-dialog')) target.closest('dialog')?.close();
    } catch (error) { notify(error.message, true); }
  });

  function filterEventPage() {
    const camera = $('event-camera-filter')?.value || 'all';
    const vehicleClass = $('event-class-filter')?.value || 'all';
    const track = ($('event-track-filter')?.value || '').trim().toLowerCase();
    let visible = 0;
    $$('#event-page-rows tr').forEach((row) => {
      const show = (camera === 'all' || row.dataset.camera === camera) && (vehicleClass === 'all' || row.dataset.class.toLowerCase() === vehicleClass.toLowerCase()) && (!track || row.dataset.track.toLowerCase().includes(track));
      row.hidden = !show;
      if (show) visible += 1;
    });
    $('event-page-empty')?.classList.toggle('hidden', visible > 0);
  }

  $('token').addEventListener('keydown', (event) => { if (event.key === 'Enter') connect(); });
  $('camera-form').addEventListener('submit', (event) => saveCamera(event).catch((error) => notify(error.message, true)));
  $('monitor-camera').addEventListener('change', async (event) => {
    state.selectedCamera = event.target.value;
    resetSnapshot();
    scheduleSnapshotStream();
    await loadRealtime().catch((error) => notify(error.message, true));
  });
  $('monitor-search').addEventListener('input', () => {
    const rows = $$('#vehicle-rows tr').map((row) => ({row, track: normalizeTrack({track_id: row.querySelector('.track-id')?.textContent || ''})}));
    if (state.activeView === 'realtime') loadRealtime().catch(() => {});
  });
  $('monitor-auto').addEventListener('change', scheduleRealtime);
  $('realtime-filter').addEventListener('submit', (event) => { event.preventDefault(); loadRealtime().catch((error) => notify(error.message, true)); });
  $('monitor-refresh').addEventListener('click', () => loadRealtime().catch((error) => notify(error.message, true)));
  $('monitor-reset').addEventListener('click', () => { $('monitor-window').value = '1h'; $('monitor-search').value = ''; loadRealtime().catch((error) => notify(error.message, true)); });

  window.addEventListener('beforeunload', () => {
    clearInterval(state.snapshotTimer);
    if (state.imageUrls.snapshot) URL.revokeObjectURL(state.imageUrls.snapshot);
  });
  syncSelectors();
  connectFromFragment().catch((error) => notify(error.message, true));
})();
