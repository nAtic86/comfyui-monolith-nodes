import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function isInputLinked(node, inputName) {
    if (!node.inputs) return false;
    const inp = node.inputs.find((i) => i.name === inputName);
    return Boolean(inp && inp.link != null);
}

function getWidgetValue(node, widgetName, fallback) {
    if (!node.widgets) return fallback;
    const w = node.widgets.find((w) => w.name === widgetName);
    return w && w.value !== undefined ? w.value : fallback;
}

function getLiveTimelineState(node) {
    const executed = node._monolithNleMeta;
    const stillDur = parseFloat(getWidgetValue(node, "still_duration", 4.0)) || 4.0;
    const endcardWidgetDur = parseFloat(getWidgetValue(node, "endcard_duration", 4.0)) || 4.0;
    const rotate = String(getWidgetValue(node, "rotate_input", "0"));
    const reverse = Boolean(getWidgetValue(node, "reverse", false));
    const hwEnc = String(getWidgetValue(node, "hw_encoder", "auto"));

    const mediaFile = String(getWidgetValue(node, "media_file", "none"));
    const overlayFile = String(getWidgetValue(node, "overlay_file", "none"));
    const alphaFile = String(getWidgetValue(node, "overlay_alpha_file", "none"));
    const endcardFile = String(getWidgetValue(node, "endcard_file", "none"));
    const audioFile = String(getWidgetValue(node, "audio_file", "none"));

    const hasMedia =
        isInputLinked(node, "media_video") ||
        isInputLinked(node, "media_image") ||
        (mediaFile && mediaFile !== "none");

    const hasOverlay =
        isInputLinked(node, "overlay_video") ||
        isInputLinked(node, "overlay_image") ||
        (overlayFile && overlayFile !== "none");

    const hasAlpha =
        isInputLinked(node, "overlay_mask") ||
        (alphaFile && alphaFile !== "none");

    const hasEndcard =
        isInputLinked(node, "endcard_video") ||
        isInputLinked(node, "endcard_image") ||
        (endcardFile && endcardFile !== "none");

    const hasExternalAudio =
        isInputLinked(node, "audio") ||
        (audioFile && audioFile !== "none");

    const mediaDur = executed ? executed.media_dur : stillDur;
    const endcardDur = hasEndcard ? (executed && executed.has_endcard ? executed.endcard_dur : endcardWidgetDur) : 0;
    const totalDur = Math.max(0.5, mediaDur + endcardDur);
    const thumbTs = Math.min(2.0, mediaDur / 2.0);

    let mediaLabel = "Media";
    if (isInputLinked(node, "media_video")) mediaLabel = "Video In";
    else if (isInputLinked(node, "media_image")) mediaLabel = "Image In";
    else if (mediaFile !== "none") mediaLabel = mediaFile;

    let overlayLabel = "Empty";
    if (isInputLinked(node, "overlay_video")) overlayLabel = "Overlay Video";
    else if (isInputLinked(node, "overlay_image")) overlayLabel = "Overlay Image";
    else if (overlayFile !== "none") overlayLabel = overlayFile;

    let endcardLabel = "None";
    if (isInputLinked(node, "endcard_video")) endcardLabel = "Endcard Video";
    else if (isInputLinked(node, "endcard_image")) endcardLabel = "Endcard Image";
    else if (endcardFile !== "none") endcardLabel = endcardFile;

    let audioLabel = isInputLinked(node, "media_video") ? "Embedded Audio" : "No Audio";
    if (hasExternalAudio) {
        audioLabel = isInputLinked(node, "audio") ? "Audio In (Backtimed)" : `${audioFile} (Backtimed)`;
    } else if (executed && executed.audio_mode === "embedded_reversed") {
        audioLabel = "Embedded Audio (Rev)";
    }

    return {
        hasMedia,
        hasOverlay,
        hasAlpha,
        hasEndcard,
        hasExternalAudio,
        mediaDur,
        endcardDur,
        totalDur,
        thumbTs,
        rotate,
        reverse,
        hwEnc: executed ? executed.hw_encoder : hwEnc,
        specs: executed ? `${executed.width}x${executed.height} @ ${executed.fps}fps` : "Pre-render Preview",
        mediaLabel,
        overlayLabel,
        endcardLabel,
        audioLabel,
    };
}

function drawNleTimeline(canvas, node) {
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(280, rect.width || (node.size ? node.size[0] - 20 : 320));
    const height = 148;

    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
        canvas.width = Math.round(width * dpr);
        canvas.height = Math.round(height * dpr);
    }

    const ctx = canvas.getContext("2d");
    ctx.save();
    ctx.scale(dpr, dpr);

    const st = getLiveTimelineState(node);

    // Background panel
    ctx.fillStyle = "#14171c";
    ctx.strokeStyle = "#2b313c";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(0.5, 0.5, width - 1, height - 1, 6);
    ctx.fill();
    ctx.stroke();

    // Header bar
    ctx.fillStyle = "#1d222b";
    ctx.beginPath();
    ctx.roundRect(1, 1, width - 2, 24, [6, 6, 0, 0]);
    ctx.fill();

    ctx.font = "600 10px Inter, system-ui, sans-serif";
    ctx.fillStyle = "#94a3b8";
    ctx.fillText("NLE TIMELINE", 8, 16);

    const badgeText = `${st.specs}  |  ${st.totalDur.toFixed(1)}s  |  ${String(st.hwEnc).toUpperCase()}`;
    ctx.font = "500 10px monospace";
    ctx.fillStyle = "#38bdf8";
    const bw = ctx.measureText(badgeText).width;
    ctx.fillText(badgeText, Math.max(92, width - bw - 8), 16);

    const labelColW = 54;
    const trackX = labelColW + 6;
    const trackW = Math.max(100, width - trackX - 10);

    // Time Ruler (y = 26..42)
    ctx.fillStyle = "#181c23";
    ctx.fillRect(1, 25, width - 2, 17);
    ctx.strokeStyle = "#262c36";
    ctx.beginPath();
    ctx.moveTo(0, 42);
    ctx.lineTo(width, 42);
    ctx.stroke();

    const stepSec = st.totalDur <= 6 ? 1 : st.totalDur <= 16 ? 2 : 5;
    ctx.font = "9px monospace";
    ctx.fillStyle = "#64748b";
    for (let t = 0; t <= st.totalDur + 0.01; t += stepSec) {
        const x = trackX + (t / st.totalDur) * trackW;
        ctx.strokeStyle = "#334155";
        ctx.beginPath();
        ctx.moveTo(x, 36);
        ctx.lineTo(x, 42);
        ctx.stroke();
        ctx.fillText(`${t.toFixed(0)}s`, Math.min(width - 20, x + 2), 35);
    }

    const mediaFrac = Math.min(1.0, st.mediaDur / st.totalDur);
    const mediaW = Math.max(16, mediaFrac * trackW);
    const endcardW = Math.max(0, trackW - mediaW);

    const tracks = [
        { id: "V2", y: 47, h: 26, title: "V2 OVR" },
        { id: "V1", y: 78, h: 28, title: "V1 VID" },
        { id: "A1", y: 111, h: 26, title: "A1 AUD" },
    ];

    for (const tr of tracks) {
        // Track label cell
        ctx.fillStyle = "#1b2028";
        ctx.fillRect(4, tr.y, labelColW - 4, tr.h);
        ctx.font = "600 9px monospace";
        ctx.fillStyle = "#94a3b8";
        ctx.fillText(tr.title, 8, tr.y + tr.h / 2 + 3);

        // Track lane background
        ctx.fillStyle = "#101317";
        ctx.strokeStyle = "#222731";
        ctx.beginPath();
        ctx.roundRect(trackX, tr.y, trackW, tr.h, 4);
        ctx.fill();
        ctx.stroke();
    }

    // Track V2: Overlay (spans 0 -> mediaDur)
    const v2 = tracks[0];
    if (st.hasOverlay) {
        ctx.fillStyle = "rgba(139, 92, 246, 0.32)";
        ctx.strokeStyle = "#8b5cf6";
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.roundRect(trackX + 1, v2.y + 2, Math.max(12, mediaW - 2), v2.h - 4, 3);
        ctx.fill();
        ctx.stroke();

        ctx.font = "500 10px Inter, system-ui, sans-serif";
        ctx.fillStyle = "#ede9fe";
        const alphaTag = st.hasAlpha ? " [+ALPHA]" : "";
        ctx.fillText(`${st.overlayLabel}${alphaTag}`, trackX + 7, v2.y + 16);
    } else {
        ctx.font = "italic 9px Inter, system-ui, sans-serif";
        ctx.fillStyle = "#475569";
        ctx.fillText("No overlay connected", trackX + 8, v2.y + 16);
    }

    // Track V1: Media (0 -> mediaDur) + Endcard (mediaDur -> totalDur)
    const v1 = tracks[1];
    ctx.fillStyle = st.hasMedia ? "rgba(14, 165, 233, 0.34)" : "rgba(100, 116, 139, 0.2)";
    ctx.strokeStyle = st.hasMedia ? "#0ea5e9" : "#475569";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.roundRect(trackX + 1, v1.y + 2, Math.max(12, mediaW - 2), v1.h - 4, 3);
    ctx.fill();
    ctx.stroke();

    const flags = [];
    if (st.reverse) flags.push("REV");
    if (st.rotate !== "0") flags.push(st.rotate);
    const flagStr = flags.length ? ` [${flags.join(",")}]` : "";
    ctx.font = "600 10px Inter, system-ui, sans-serif";
    ctx.fillStyle = st.hasMedia ? "#e0f2fe" : "#94a3b8";
    ctx.fillText(`${st.mediaLabel}${flagStr} (${st.mediaDur.toFixed(1)}s)`, trackX + 7, v1.y + 17);

    if (st.hasEndcard && endcardW > 6) {
        ctx.fillStyle = "rgba(245, 158, 11, 0.32)";
        ctx.strokeStyle = "#f59e0b";
        ctx.beginPath();
        ctx.roundRect(trackX + mediaW + 1, v1.y + 2, Math.max(8, endcardW - 2), v1.h - 4, 3);
        ctx.fill();
        ctx.stroke();

        ctx.font = "600 10px Inter, system-ui, sans-serif";
        ctx.fillStyle = "#fef3c7";
        ctx.fillText(`Endcard (${st.endcardDur.toFixed(1)}s)`, trackX + mediaW + 6, v1.y + 17);
    }

    // Track A1: Audio (Backtimed across totalDur or Embedded)
    const a1 = tracks[2];
    const audioActive = st.hasExternalAudio || isInputLinked(node, "media_video");
    const audioSpanW = st.hasExternalAudio ? trackW : mediaW;
    if (audioActive) {
        ctx.fillStyle = st.hasExternalAudio ? "rgba(16, 185, 129, 0.30)" : "rgba(20, 184, 166, 0.22)";
        ctx.strokeStyle = st.hasExternalAudio ? "#10b981" : "#14b8a6";
        ctx.beginPath();
        ctx.roundRect(trackX + 1, a1.y + 2, Math.max(12, audioSpanW - 2), a1.h - 4, 3);
        ctx.fill();
        ctx.stroke();

        // Draw 6-frame fade-in diagonal ramp on external audio
        if (st.hasExternalAudio) {
            ctx.strokeStyle = "#6ee7b7";
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(trackX + 2, a1.y + a1.h - 3);
            ctx.lineTo(trackX + 18, a1.y + 3);
            ctx.stroke();
        }

        ctx.font = "500 10px Inter, system-ui, sans-serif";
        ctx.fillStyle = "#d1fae5";
        ctx.fillText(st.audioLabel, trackX + (st.hasExternalAudio ? 22 : 7), a1.y + 16);
    } else {
        ctx.font = "italic 9px Inter, system-ui, sans-serif";
        ctx.fillStyle = "#475569";
        ctx.fillText("Muted / No audio source", trackX + 8, a1.y + 16);
    }

    // Midpoint thumbnail extraction marker (playhead line)
    const thumbX = trackX + Math.min(1.0, st.thumbTs / st.totalDur) * trackW;
    ctx.strokeStyle = "#f43f5e";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 2]);
    ctx.beginPath();
    ctx.moveTo(thumbX, 38);
    ctx.lineTo(thumbX, height - 6);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = "#f43f5e";
    ctx.beginPath();
    ctx.moveTo(thumbX - 4, 38);
    ctx.lineTo(thumbX + 4, 38);
    ctx.lineTo(thumbX, 43);
    ctx.closePath();
    ctx.fill();

    ctx.restore();
}

function attachVideoPlayerAndTimeline(node, includeTimeline) {
    const container = document.createElement("div");
    container.style.display = "flex";
    container.style.flexDirection = "column";
    container.style.gap = "6px";
    container.style.width = "100%";
    container.style.boxSizing = "border-box";
    container.style.padding = "2px 0";

    let canvas = null;
    if (includeTimeline) {
        canvas = document.createElement("canvas");
        canvas.style.width = "100%";
        canvas.style.height = "148px";
        canvas.style.display = "block";
        canvas.style.borderRadius = "6px";
        container.appendChild(canvas);
    }

    const videoEl = document.createElement("video");
    videoEl.controls = true;
    videoEl.loop = true;
    videoEl.muted = true;
    videoEl.autoplay = true;
    videoEl.playsInline = true;
    videoEl.style.width = "100%";
    videoEl.style.maxHeight = "260px";
    videoEl.style.borderRadius = "6px";
    videoEl.style.background = "#090b0e";
    videoEl.style.display = "none";
    container.appendChild(videoEl);

    const domWidget = node.addDOMWidget("monolith_nle_view", "custom", container, {
        serialize: false,
        hideOnZoom: false,
    });

    domWidget.computeSize = function (width) {
        const timelineH = includeTimeline ? 156 : 0;
        const videoH = videoEl.style.display !== "none" ? 220 : 0;
        return [width, timelineH + videoH];
    };

    const refresh = () => {
        if (canvas) {
            drawNleTimeline(canvas, node);
        }
    };

    if (includeTimeline) {
        setTimeout(refresh, 50);
        const origConn = node.onConnectionsChange;
        node.onConnectionsChange = function () {
            if (origConn) origConn.apply(this, arguments);
            refresh();
        };
        const origResize = node.onResize;
        node.onResize = function () {
            if (origResize) origResize.apply(this, arguments);
            refresh();
        };
        if (node.widgets) {
            for (const w of node.widgets) {
                const origCb = w.callback;
                w.callback = function () {
                    if (origCb) origCb.apply(this, arguments);
                    refresh();
                };
            }
        }
    }

    const origExecuted = node.onExecuted;
    node.onExecuted = function (output) {
        if (origExecuted) origExecuted.apply(this, arguments);
        if (output && output.nle_timeline && output.nle_timeline[0]) {
            node._monolithNleMeta = output.nle_timeline[0];
        }
        const vidInfo = (output && output.videos && output.videos[0]) || (output && output.images && output.images[0]);
        if (vidInfo && vidInfo.filename && String(vidInfo.filename).endsWith(".mp4")) {
            const params = new URLSearchParams({
                filename: vidInfo.filename,
                subfolder: vidInfo.subfolder || "",
                type: vidInfo.type || "output",
                t: String(Date.now()),
            });
            videoEl.src = api.apiURL(`/view?${params.toString()}`);
            videoEl.style.display = "block";
            if (node.setSize && node.computeSize) {
                const sz = node.computeSize();
                node.setSize([Math.max(node.size[0], sz[0]), Math.max(node.size[1], sz[1])]);
            }
        }
        refresh();
    };
}

app.registerExtension({
    name: "Monolith.NLE",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (
            nodeData.name === "MonolithNLEAssemblerV1" ||
            nodeData.name === "MonolithNLEAssemblerV2" ||
            nodeData.name === "MonolithNLEAssemblerV3"
        ) {
            const includeTimeline = nodeData.name === "MonolithNLEAssemblerV2";
            const origCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (origCreated) origCreated.apply(this, arguments);
                if (includeTimeline && this.size) {
                    this.size[0] = Math.max(this.size[0], 360);
                }
                attachVideoPlayerAndTimeline(this, includeTimeline);
            };
        }
    },
});
