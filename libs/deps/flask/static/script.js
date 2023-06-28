function popup(title, message, container_elem) {
    const el = $(`<div style="width: ${$(container_elem).width() - 2* parseFloat($("body").css("font-size"))}px", class="popup-container">
    <button class="close-button" aria-label="Close alert" type="button" data-close>
        <span aria-hidden="true">&times;</span>
    </button>
    <h2>${title}</h2>
    <h3>${message}</h3>
    </div>`);

    $(container_elem).append(el)
    
    el.on("pointerup", function(event) {
        $(event.target).closest(".popup-container").remove()
    })
    
}

$(document).ajaxError(function myErrorHandler(
    event,
    xhr,
    ajaxOptions,
    thrownError
) {
    console.log(event);
    console.log(ajaxOptions);
    console.log(thrownError);
    const err = xhr.responseText;
    console.log(err);

    const container = $(".container")[0]
    console.log($(container).width())
    popup("An error has occured", err, container)

    // alert(err);
});
