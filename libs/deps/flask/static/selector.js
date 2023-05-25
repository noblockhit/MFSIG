$(window).on("load", function () {
    $.get("/cameras", function (data) {
        $("#cam-select").html(
            `<option value="-1" selected disabled hidden>No camera selected...</option>` +
                data
        );
        adjust_option_width();
    });

    $("#cam-select").on("change", function () {
        $.ajax({
            url: "/camera/" + this.value,
            type: "POST",
            dataType: "text",
            success: function (data) {
                console.log("success " + data);
                $("#reso-select").html(
                    `<option value="-1" selected disabled hidden>No resolution selected...</option>` +
                        data
                );
                adjust_option_width();
            },

            error: function (err) {
                console.log("Ajax error:");
                console.log(err);
            },
        });
    });

    $("#reso-select").on("change", function () {
        $.ajax({
            url: "/resolution/" + this.value,
            type: "POST",
            success: function (data) {
                $("#live-view-container").removeAttr('hidden');
            },
            error: function (err) {
                console.log("Ajax error:");
                console.log(err);
            },
        });
        adjust_option_width();
    });

    adjust_option_width();
});

function adjust_option_width() {
    var e = document.querySelectorAll("option");
    e.forEach((x) => {
        if (x.textContent.length > 20)
            x.textContent = x.textContent.substring(0, 20) + "...";
    });
}

window.onresize = adjust_option_width;
