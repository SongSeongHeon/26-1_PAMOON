/*
 * Copyright 2023 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0
 */
package com.samsung.android.ecgmonitor;

import static android.content.pm.PackageManager.PERMISSION_DENIED;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;

import com.samsung.android.ecgmonitor.databinding.ActivityMainBinding;
import com.samsung.android.service.health.tracking.ConnectionListener;
import com.samsung.android.service.health.tracking.HealthTracker;
import com.samsung.android.service.health.tracking.HealthTrackerException;
import com.samsung.android.service.health.tracking.HealthTrackingService;
import com.samsung.android.service.health.tracking.data.DataPoint;
import com.samsung.android.service.health.tracking.data.HealthTrackerType;
import com.samsung.android.service.health.tracking.data.ValueKey;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public class MainActivity extends Activity {

    private static final String APP_TAG = "ECG Auth Collector";
    private static final String FLASK_RECEIVE_URL =
            "http://YOUR_SERVER_IP:5000/api/ecg-json/receive";

    private static final int ANDROID_16_API_LEVEL = 36;

    private String permission;

    private final Handler ecgHandler = new Handler(Looper.getMainLooper());

    private final AtomicBoolean isMeasurementRunning = new AtomicBoolean(false);
    private final AtomicBoolean leadOff = new AtomicBoolean(true);
    private final AtomicReference<Float> curEcg = new AtomicReference<>(0.0f);

    private final AtomicInteger leadOffCount = new AtomicInteger(0);
    private final AtomicInteger receivedBatchCount = new AtomicInteger(0);
    private final AtomicInteger totalSampleCount = new AtomicInteger(0);
    private final AtomicInteger validSampleCount = new AtomicInteger(0);
    private final AtomicInteger invalidSampleCount = new AtomicInteger(0);
    private final AtomicInteger warmupSkipCount = new AtomicInteger(0);

    private final List<Float> ecgSamples = new ArrayList<>();

    private static final int MEASUREMENT_DURATION = 30000;
    private static final int MEASUREMENT_TICK = 1000;
    private static final int SAMPLING_RATE = 500;
    private static final int NO_CONTACT = 5;

    private static final int WARMUP_SAMPLE_SKIP = 500;
    private static final int MIN_VALID_SAMPLE_COUNT = 10000;

    private static final float ECG_MIN_VALID = -150.0f;
    private static final float ECG_MAX_VALID = 150.0f;

    private static final String SUBJECT_ID = "USER-01";
    private static final String SUBJECT_NAME = "ECG 데이터";
    private static final String SUBJECT_BIRTH_DATE = "20020201";
    private static final String DEVICE_NAME = "Samsung Galaxy Watch 6";

    private boolean permissionGranted = false;
    private boolean connected = false;

    private TextView mTextView;
    private Button mButMeasure;
    private ActivityMainBinding binding;

    private HealthTrackingService healthTrackingService = null;
    private HealthTracker ecgTracker = null;
    private CountDownTimer countDownTimer = null;

    private final HealthTracker.TrackerEventListener ecgListener =
            new HealthTracker.TrackerEventListener() {
                @Override
                public void onDataReceived(@NonNull List<DataPoint> list) {
                    if (list.isEmpty()) {
                        return;
                    }

                    boolean batchHasValidContact = false;

                    for (DataPoint dataPoint : list) {
                        int isLeadOff = dataPoint.getValue(ValueKey.EcgSet.LEAD_OFF);

                        if (isLeadOff == NO_CONTACT) {
                            leadOff.set(true);
                            leadOffCount.incrementAndGet();
                            continue;
                        }

                        batchHasValidContact = true;
                        leadOff.set(false);

                        float ecgMv = dataPoint.getValue(ValueKey.EcgSet.ECG_MV);
                        totalSampleCount.incrementAndGet();

                        if (!isUsableEcgValue(ecgMv)) {
                            invalidSampleCount.incrementAndGet();
                            continue;
                        }

                        if (warmupSkipCount.get() < WARMUP_SAMPLE_SKIP) {
                            warmupSkipCount.incrementAndGet();
                            continue;
                        }

                        synchronized (ecgSamples) {
                            ecgSamples.add(ecgMv);
                        }

                        validSampleCount.incrementAndGet();
                        curEcg.set(ecgMv);
                    }

                    if (batchHasValidContact) {
                        receivedBatchCount.incrementAndGet();
                    }
                }

                @Override
                public void onFlushCompleted() {
                    Log.i(APP_TAG, "onFlushCompleted called");
                }

                @Override
                public void onError(HealthTracker.TrackerError trackerError) {
                    Log.i(APP_TAG, "onError called: " + trackerError);

                    if (trackerError == HealthTracker.TrackerError.PERMISSION_ERROR) {
                        runOnUiThread(() ->
                                Toast.makeText(
                                        getApplicationContext(),
                                        getString(R.string.NoPermission),
                                        Toast.LENGTH_SHORT
                                ).show()
                        );
                    }

                    if (trackerError == HealthTracker.TrackerError.SDK_POLICY_ERROR) {
                        runOnUiThread(() ->
                                Toast.makeText(
                                        getApplicationContext(),
                                        getString(R.string.SDKPolicyError),
                                        Toast.LENGTH_SHORT
                                ).show()
                        );
                    }
                }
            };

    private final ConnectionListener connectionListener = new ConnectionListener() {
        @Override
        public void onConnectionSuccess() {
            Log.i(APP_TAG, "Connected to Health Tracking Service");

            Toast.makeText(
                    getApplicationContext(),
                    getString(R.string.ConnectedToHS),
                    Toast.LENGTH_SHORT
            ).show();

            checkCapabilities();

            connected = true;
            ecgTracker = healthTrackingService.getHealthTracker(HealthTrackerType.ECG_ON_DEMAND);
        }

        @Override
        public void onConnectionEnded() {
            Log.i(APP_TAG, "Disconnected from Health Tracking Service");
            connected = false;
        }

        @Override
        public void onConnectionFailed(HealthTrackerException e) {
            if (e.getErrorCode() == HealthTrackerException.OLD_PLATFORM_VERSION
                    || e.getErrorCode() == HealthTrackerException.PACKAGE_NOT_INSTALLED) {
                Toast.makeText(
                        getApplicationContext(),
                        getString(R.string.NoHealthPlatformError),
                        Toast.LENGTH_LONG
                ).show();
            }

            if (e.hasResolution()) {
                e.resolve(MainActivity.this);
            } else {
                Log.e(APP_TAG, "Could not connect to Health Services: " + e.getMessage());

                runOnUiThread(() ->
                        Toast.makeText(
                                getApplicationContext(),
                                getString(R.string.ConnectionError),
                                Toast.LENGTH_LONG
                        ).show()
                );
            }

            finish();
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        mTextView = binding.txtOutput;
        mButMeasure = binding.butStart;

        mButMeasure.setOnClickListener(unused -> startMeasurement());

        if (Build.VERSION.SDK_INT >= ANDROID_16_API_LEVEL) {
            permission = getString(R.string.additionalHealthDataPermission);
        } else {
            permission = Manifest.permission.BODY_SENSORS;
        }

        if (ActivityCompat.checkSelfPermission(
                getApplicationContext(),
                permission
        ) == PackageManager.PERMISSION_DENIED) {
            requestPermissions(new String[]{permission}, 0);
        } else {
            permissionGranted = true;
        }

        try {
            healthTrackingService = new HealthTrackingService(
                    connectionListener,
                    getApplicationContext()
            );
            healthTrackingService.connectService();
        } catch (Throwable t) {
            final String msg = t.getMessage();
            Log.e(APP_TAG, msg == null ? "" : msg);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        stopTrackerSafely();

        isMeasurementRunning.set(false);
        ecgHandler.removeCallbacksAndMessages(null);

        if (countDownTimer != null) {
            countDownTimer.cancel();
            countDownTimer = null;
        }

        if (healthTrackingService != null) {
            healthTrackingService.disconnectService();
        }
    }

    private void checkCapabilities() {
        final List<HealthTrackerType> availableTrackers =
                healthTrackingService
                        .getTrackingCapability()
                        .getSupportHealthTrackerTypes();

        if (!availableTrackers.contains(HealthTrackerType.ECG_ON_DEMAND)) {
            Toast.makeText(
                    getApplicationContext(),
                    getString(R.string.NoECGSupport),
                    Toast.LENGTH_LONG
            ).show();

            Log.e(APP_TAG, "Device does not support ECG tracking");
            finish();
        }
    }

    private void startMeasurement() {
        if (ActivityCompat.checkSelfPermission(
                getApplicationContext(),
                permission
        ) == PackageManager.PERMISSION_DENIED) {
            requestPermissions(new String[]{permission}, 0);
        }

        if (!permissionGranted) {
            Log.i(APP_TAG, "Could not get permissions. Terminating measurement");
            return;
        }

        if (!connected || ecgTracker == null) {
            Toast.makeText(
                    getApplicationContext(),
                    getString(R.string.ConnectionError),
                    Toast.LENGTH_SHORT
            ).show();
            return;
        }

        if (!isMeasurementRunning.get()) {
            startEcgCollection();
        } else {
            stopEcgCollectionByUser();
        }
    }

    private void startEcgCollection() {
        resetMeasurementBuffers();

        mTextView.setText("ECG 측정 중...");
        mButMeasure.setText(R.string.stop);

        isMeasurementRunning.set(true);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        ecgHandler.post(() -> ecgTracker.setEventListener(ecgListener));

        countDownTimer = createMeasurementTimer();
        countDownTimer.start();
    }

    private CountDownTimer createMeasurementTimer() {
        return new CountDownTimer(MEASUREMENT_DURATION, MEASUREMENT_TICK) {
            @Override
            public void onTick(long timeLeft) {
                if (!isMeasurementRunning.get()) {
                    return;
                }

                int sampleCount;

                synchronized (ecgSamples) {
                    sampleCount = ecgSamples.size();
                }

                if (leadOff.get()) {
                    runOnUiThread(() ->
                            binding.txtOutput.setText(
                                    "전극 접촉 불안정\n"
                                            + "손가락을 유지하세요\n\n"
                                            + "남은 시간: " + (timeLeft / 1000) + "초\n"
                            )
                    );
                } else {
                    final String measureValue = String.format(
                            Locale.KOREA,
                            "ECG 측정 중\n남은 시간: %d초\n현재 ECG: %.3f mV",
                            timeLeft / 1000,
                            curEcg.get()
                    );

                    runOnUiThread(() -> binding.txtOutput.setText(measureValue));
                }
            }

            @Override
            public void onFinish() {
                finishEcgCollection();
            }
        };
    }

    private void finishEcgCollection() {
        stopTrackerSafely();

        isMeasurementRunning.set(false);

        runOnUiThread(() -> {
            getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

            int sampleCount;

            synchronized (ecgSamples) {
                sampleCount = ecgSamples.size();
            }

            if (sampleCount < MIN_VALID_SAMPLE_COUNT) {
                binding.txtOutput.setText(
                        "ECG 측정 실패\n"
                                + "유효 샘플 부족\n\n"
                                + "손목 착용 상태와\n"
                                + "손가락 접촉을 확인하세요\n\n"
                                + "유효 샘플: " + sampleCount
                );

                binding.butStart.setText(R.string.RepeatMeasurement);
                return;
            }

            binding.txtOutput.setText(
                    "ECG 측정 완료\n(PC 전송 중...)\n\n유효 샘플: " + sampleCount
            );
            binding.butStart.setText(R.string.RepeatMeasurement);

            Thread sendThread = new Thread(() -> {
                File savedFile = saveEcgSamplesAsJson();
                boolean sentToPc = sendEcgJsonToPc(savedFile);

                String sendStatus;

                if (savedFile == null) {
                    sendStatus = "JSON 저장 실패";
                } else if (sentToPc) {
                    sendStatus = "PC 전송 성공";
                } else {
                    sendStatus = "PC 전송 실패";
                }

                runOnUiThread(() -> {
                    String resultText = String.format(
                            Locale.KOREA,
                            "ECG 측정 완료\n(%s)\n\n유효 샘플: %d",
                            sendStatus,
                            sampleCount
                    );

                    binding.txtOutput.setText(resultText);
                    binding.butStart.setText(R.string.RepeatMeasurement);
                });
            });

            sendThread.start();
        });
    }

    private void stopEcgCollectionByUser() {
        stopTrackerSafely();

        if (countDownTimer != null) {
            countDownTimer.cancel();
            countDownTimer = null;
        }

        isMeasurementRunning.set(false);
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        mButMeasure.setText(R.string.start);
        mTextView.setText("ECG 측정이 중지되었습니다.");
    }

    private void stopTrackerSafely() {
        try {
            if (ecgTracker != null) {
                ecgTracker.unsetEventListener();
            }
        } catch (Exception error) {
            Log.e(APP_TAG, "Failed to unset ECG listener: " + error.getMessage());
        }
    }

    private void resetMeasurementBuffers() {
        synchronized (ecgSamples) {
            ecgSamples.clear();
        }

        curEcg.set(0.0f);
        leadOff.set(true);
        leadOffCount.set(0);
        receivedBatchCount.set(0);
        totalSampleCount.set(0);
        validSampleCount.set(0);
        invalidSampleCount.set(0);
        warmupSkipCount.set(0);
    }

    private File saveEcgSamplesAsJson() {
        try {
            JSONObject root = new JSONObject();
            JSONArray ecgArray = new JSONArray();

            int sampleCount;

            synchronized (ecgSamples) {
                sampleCount = ecgSamples.size();

                for (Float value : ecgSamples) {
                    ecgArray.put(value);
                }
            }

            String measuredAt = getCurrentTimestamp();

            root.put("name", SUBJECT_NAME);
            root.put("subject_id", SUBJECT_ID);
            root.put("birth_date", SUBJECT_BIRTH_DATE);
            root.put("source", "Samsung Health Sensor SDK");
            root.put("tracker_type", "ECG_ON_DEMAND");
            root.put("device", DEVICE_NAME);
            root.put("sampling_rate", SAMPLING_RATE);
            root.put("duration_ms", MEASUREMENT_DURATION);
            root.put("duration_seconds", sampleCount / (double) SAMPLING_RATE);
            root.put("unit", "mV");
            root.put("lead", "Lead I ECG");
            root.put("measured_at", measuredAt);

            root.put("sample_count", sampleCount);
            root.put("total_sample_count", totalSampleCount.get());
            root.put("valid_sample_count", validSampleCount.get());
            root.put("invalid_sample_count", invalidSampleCount.get());
            root.put("warmup_skip_count", warmupSkipCount.get());
            root.put("lead_off_count", leadOffCount.get());
            root.put("received_batch_count", receivedBatchCount.get());
            root.put("quality_status", getMeasurementQualityStatus(sampleCount));

            root.put("ecg_mv", ecgArray);

            String fileName = SUBJECT_ID
                    + "_"
                    + SUBJECT_BIRTH_DATE
                    + "_"
                    + getFileTimestamp()
                    + ".json";

            root.put("filename", fileName);

            File outputFile = new File(getFilesDir(), fileName);

            FileOutputStream fos = new FileOutputStream(outputFile);
            fos.write(root.toString(2).getBytes(StandardCharsets.UTF_8));
            fos.close();

            Log.i(APP_TAG, "Saved ECG JSON: " + outputFile.getAbsolutePath());
            Log.i(APP_TAG, "Valid sample count: " + sampleCount);
            Log.i(APP_TAG, "Lead-off count: " + leadOffCount.get());

            return outputFile;

        } catch (Exception error) {
            Log.e(APP_TAG, "Failed to save ECG JSON: " + error.getMessage());
            return null;
        }
    }

    private boolean sendEcgJsonToPc(File jsonFile) {
        if (jsonFile == null || !jsonFile.exists()) {
            Log.e(APP_TAG, "ECG JSON file is missing.");
            return false;
        }

        HttpURLConnection connection = null;

        try {
            String jsonBody = new String(
                    Files.readAllBytes(jsonFile.toPath()),
                    StandardCharsets.UTF_8
            );

            URL url = new URL(FLASK_RECEIVE_URL);
            connection = (HttpURLConnection) url.openConnection();

            connection.setRequestMethod("POST");
            connection.setConnectTimeout(8000);
            connection.setReadTimeout(20000);
            connection.setDoOutput(true);
            connection.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
            connection.setRequestProperty("Accept", "application/json");
            connection.setRequestProperty("X-ECG-Filename", jsonFile.getName());

            byte[] bodyBytes = jsonBody.getBytes(StandardCharsets.UTF_8);
            connection.setFixedLengthStreamingMode(bodyBytes.length);

            try (OutputStream outputStream = connection.getOutputStream()) {
                outputStream.write(bodyBytes);
                outputStream.flush();
            }

            int responseCode = connection.getResponseCode();
            Log.i(APP_TAG, "Flask response code: " + responseCode);

            return responseCode >= 200 && responseCode < 300;

        } catch (Exception error) {
            Log.e(APP_TAG, "Failed to send ECG JSON to PC: " + error.getMessage());
            return false;
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    private boolean isUsableEcgValue(float value) {
        if (Float.isNaN(value) || Float.isInfinite(value)) {
            return false;
        }

        return !(value <= ECG_MIN_VALID || value >= ECG_MAX_VALID);
    }

    private String getMeasurementQualityStatus(int sampleCount) {
        if (sampleCount >= 14000 && leadOffCount.get() <= 20) {
            return "stable";
        }

        if (sampleCount >= MIN_VALID_SAMPLE_COUNT) {
            return "caution";
        }

        return "warning";
    }

    private String getCurrentTimestamp() {
        return new SimpleDateFormat(
                "yyyy-MM-dd HH:mm:ss",
                Locale.KOREA
        ).format(new Date());
    }

    private String getFileTimestamp() {
        return new SimpleDateFormat(
                "yyyyMMdd_HHmmss",
                Locale.KOREA
        ).format(new Date());
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode,
            @NonNull String[] permissions,
            @NonNull int[] grantResults
    ) {
        if (requestCode == 0) {
            permissionGranted = true;

            for (int i = 0; i < permissions.length; ++i) {
                if (grantResults[i] == PERMISSION_DENIED) {
                    if (!shouldShowRequestPermissionRationale(permissions[i])) {
                        Toast.makeText(
                                getApplicationContext(),
                                getString(R.string.PermissionDeniedPermanently),
                                Toast.LENGTH_LONG
                        ).show();
                    } else {
                        Toast.makeText(
                                getApplicationContext(),
                                getString(R.string.PermissionDeniedRationale),
                                Toast.LENGTH_LONG
                        ).show();
                    }

                    permissionGranted = false;
                    break;
                }
            }
        }

        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }
}